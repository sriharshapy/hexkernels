/* HMX int8 GQA QK^T (bit-exact) -- the ACCELERATED expert.
 * Identical HMX tiling/crouton flow as the single-head i8_attention_qkt_hmx
 * sibling (2x2 grid of 32x32 output tiles per head, each accumulating over 2
 * D-tiles, weight packed straight from K's natural [S,D] layout with swapped
 * tile indices), but exploits the GQA structure: the shared KV head's 4
 * weight croutons (one per (tj,kt) output-col/reduction tile pair) are
 * packed into VTCM ONCE, up front, and left RESIDENT there -- the GROUP_SIZE
 * query heads that share this KV head reuse them without re-packing (only
 * the activation/Q side is re-packed per head/tile). This removes half the
 * weight-pack work (the expensive scalar-crouton-write + bulk-copy step)
 * relative to a naive per-head re-pack.
 *
 * PERF: scalar accesses to VTCM cost ~48 cycles each in timing mode, so all
 * crouton pack/unpack is STAGED through cacheable DDR buffers and moved
 * to/from VTCM in bulk 128B vector copies.
 * Recipe: docs/superpowers/specs/2026-06-12-hmx-int8-layout-re-agenda.md */
#include "kernel_api.h"
#include "harness_common.h"
#include <hexagon_types.h>

#define T  32
#define NT (GQA_S / T)   /* tiles per axis (2 for S=D=64) */

void candidate_kernel(const uint8_t *Q, const int8_t *K, uint16_t *out) {
    uint8_t          *vAct  = (uint8_t  *)(uintptr_t)(HVX_VTCM_BASE + 0x0000u); /* 2048B, re-packed per (head,tile) */
    /* NT*NT persistent weight-crouton tiles for the single shared KV head, 1024B each */
    int8_t           *vWgtTiles[NT][NT];
    volatile uint8_t *vBias = (volatile uint8_t *)(uintptr_t)(HVX_VTCM_BASE + 0x2000u);
    uint16_t         *vOut  = (uint16_t *)(uintptr_t)(HVX_VTCM_BASE + 0x3000u);
    for (int tj = 0; tj < NT; tj++)
        for (int kt = 0; kt < NT; kt++)
            vWgtTiles[tj][kt] = (int8_t *)(uintptr_t)(HVX_VTCM_BASE + 0x1000u + (unsigned)(tj*NT + kt) * 0x400u);

    static uint8_t  aAct[2048] HVX_ALIGN;   /* cacheable crouton staging */
    static int8_t   aWgt[1024] HVX_ALIGN;
    static uint16_t aOut[T*T]  HVX_ALIGN;

    for (int i = 0; i < 2048; i++) aAct[i]  = 0;    /* even/low bytes stay 0 */
    for (int i = 0; i < 2048; i++) vBias[i] = 0x40; /* requant: scale 17/16, bias 0 */

    const unsigned lim_a = 2047, lim_w = 1023, lim_o = 2047;

    /* Pack the shared KV head's weight croutons ONCE (h_kv = 0, the only KV head). */
    const int8_t *K0 = K; /* K[0] -- GQA_H_KV == 1 */
    for (int tj = 0; tj < NT; tj++) {
        for (int kt = 0; kt < NT; kt++) {
            for (int k = 0; k < T; k++)
                for (int j = 0; j < T; j++)
                    /* weight[k][j] = K^T[k][j] = K0[j][k]: swapped tile indices,
                     * same trick as the single-head QK^T sibling task */
                    aWgt[hvx_hmx_i8_wgt_off(k, j)] = K0[(tj*T + j)*GQA_D + (kt*T + k)];
            { HVX_Vector *s = (HVX_Vector *)aWgt, *d = (HVX_Vector *)vWgtTiles[tj][kt];
              for (int b = 0; b < 8; b++) d[b] = s[b]; }
        }
    }

    for (int hq = 0; hq < GQA_H_Q; hq++) {
        const uint8_t *Qh = Q + (size_t)hq * GQA_S * GQA_D;
        uint16_t      *Oh = out + (size_t)hq * GQA_S * GQA_S;
        for (int ti = 0; ti < NT; ti++) {          /* query-row tile */
            for (int tj = 0; tj < NT; tj++) {      /* key-row tile (= output column tile) */
                __asm__ volatile("mxclracc\n");                 /* clear accumulator per out tile */
                for (int kt = 0; kt < NT; kt++) {  /* D-axis (reduction) tile */
                    /* only the activation (Q) side is re-packed -- weight is already
                     * resident in VTCM from the pre-pass above, shared across heads */
                    for (int i = 0; i < T; i++)
                        for (int k = 0; k < T; k++)
                            aAct[hvx_hmx_i8_act_off(i, k)] = Qh[(ti*T + i)*GQA_D + (kt*T + k)];
                    { HVX_Vector *s = (HVX_Vector *)aAct, *d = (HVX_Vector *)vAct;
                      for (int b = 0; b < 16; b++) d[b] = s[b]; }
                    __asm__ volatile("{ activation.ub=mxmem(%0,%1)\n\t weight.b=mxmem(%2,%3) }\n"
                                     :: "r"(vAct), "r"(lim_a), "r"(vWgtTiles[tj][kt]), "r"(lim_w) : "memory");
                }
                __asm__ volatile("bias=mxmem(%0)\n" :: "r"(vBias) : "memory");
                __asm__ volatile("mxmem(%0,%1):after.uh=acc:2x1\n" :: "r"(vOut), "r"(lim_o) : "memory");
                __asm__ volatile("isync\n\t");
                { HVX_Vector *s = (HVX_Vector *)vOut, *d = (HVX_Vector *)aOut;
                  for (int b = 0; b < 16; b++) d[b] = s[b]; }
                for (int i = 0; i < T; i++)
                    for (int j = 0; j < T; j++)
                        Oh[(ti*T + i)*GQA_S + (tj*T + j)] = aOut[hvx_hmx_i8_out_off(i, j)];
            }
        }
    }
}
