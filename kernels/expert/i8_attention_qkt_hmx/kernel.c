/* HMX int8 attention QK^T (bit-exact) -- the ACCELERATED expert.
 * S=64 is a 2x2 grid of 32x32 output tiles; each output tile accumulates over
 * two D-tiles (D=64 = 2*32) using the proven 32x32 crouton packing (decoded
 * 2026-06-12) issued in a tiling loop -- identical mechanism to
 * i8_matmul_hmx_64x64, but the WEIGHT (K) is packed straight from its natural
 * [S,D] row-major layout with swapped tile indices (weight[k][j] = K[j][d] =
 * K[(tj*T+j)*D + (kt*T+k)]) so no separate transpose-staging buffer is needed
 * -- attention's K is already "pre-transposed" relative to a generic A.B
 * matmul's B operand. Activation lives in the HIGH byte of an fp16-crouton
 * slot (2048B tile, lim 2047), weight is 4-deep packed (1024B tile, lim 1023)
 * -- the two operands need DIFFERENT load limits in one packet -> explicit
 * asm. The 0x40-config store applies the (acc*17+8)>>4 & 0xFFF requant.
 *
 * PERF: scalar accesses to VTCM cost ~48 cycles each in timing mode, so the
 * crouton pack/unpack is STAGED through cacheable DDR buffers and moved to/from
 * VTCM in bulk 128B vector copies (a software analog of the DMA an on-target
 * kernel would use). This is what turns the raw HMX matmul win into a real
 * end-to-end win over the HVX vrmpy baseline.
 * Recipe: docs/superpowers/specs/2026-06-12-hmx-int8-layout-re-agenda.md */
#include "kernel_api.h"
#include "harness_common.h"
#include <hexagon_types.h>

void candidate_kernel(const uint8_t *Q, const int8_t *K, uint16_t *out, int S, int D) {
    uint8_t          *vAct  = (uint8_t  *)(uintptr_t)(HVX_VTCM_BASE + 0x0000u); /* 2048B */
    int8_t           *vWgt  = (int8_t   *)(uintptr_t)(HVX_VTCM_BASE + 0x1000u); /* 1024B */
    volatile uint8_t *vBias = (volatile uint8_t *)(uintptr_t)(HVX_VTCM_BASE + 0x2000u);
    uint16_t         *vOut  = (uint16_t *)(uintptr_t)(HVX_VTCM_BASE + 0x3000u);

    static uint8_t  aAct[2048]  HVX_ALIGN;   /* cacheable crouton staging */
    static int8_t   aWgt[1024]  HVX_ALIGN;
    static uint16_t aOut[32*32] HVX_ALIGN;

    for (int i = 0; i < 2048; i++) aAct[i]  = 0;    /* even/low bytes stay 0 */
    for (int i = 0; i < 2048; i++) vBias[i] = 0x40; /* requant: scale 17/16, bias 0 */

    const unsigned lim_a = 2047, lim_w = 1023, lim_o = 2047;
    const int T = 32;             /* crouton tile edge */
    const int nt = S / T;         /* tiles per axis (2 for S=D=64) */

    for (int ti = 0; ti < nt; ti++) {          /* query-row tile */
        for (int tj = 0; tj < nt; tj++) {      /* key-row tile (= output column tile) */
            __asm__ volatile("mxclracc\n");                 /* clear accumulator per out tile */
            for (int kt = 0; kt < nt; kt++) {  /* D-axis (reduction) tile */
                for (int i = 0; i < T; i++)
                    for (int k = 0; k < T; k++)
                        aAct[hvx_hmx_i8_act_off(i, k)] = Q[(ti*T + i)*D + (kt*T + k)];
                for (int k = 0; k < T; k++)
                    for (int j = 0; j < T; j++)
                        /* weight[k][j] = K^T[k][j] = K[j][k]: read K directly with
                         * swapped tile indices -- no transpose buffer needed */
                        aWgt[hvx_hmx_i8_wgt_off(k, j)] = K[(tj*T + j)*D + (kt*T + k)];
                { HVX_Vector *s = (HVX_Vector *)aAct, *d = (HVX_Vector *)vAct;
                  for (int b = 0; b < 16; b++) d[b] = s[b]; }
                { HVX_Vector *s = (HVX_Vector *)aWgt, *d = (HVX_Vector *)vWgt;
                  for (int b = 0; b < 8; b++) d[b] = s[b]; }
                __asm__ volatile("{ activation.ub=mxmem(%0,%1)\n\t weight.b=mxmem(%2,%3) }\n"
                                 :: "r"(vAct), "r"(lim_a), "r"(vWgt), "r"(lim_w) : "memory");
            }
            __asm__ volatile("bias=mxmem(%0)\n" :: "r"(vBias) : "memory");
            __asm__ volatile("mxmem(%0,%1):after.uh=acc:2x1\n" :: "r"(vOut), "r"(lim_o) : "memory");
            __asm__ volatile("isync\n\t");
            { HVX_Vector *s = (HVX_Vector *)vOut, *d = (HVX_Vector *)aOut;
              for (int b = 0; b < 16; b++) d[b] = s[b]; }
            for (int i = 0; i < T; i++)
                for (int j = 0; j < T; j++)
                    out[(ti*T + i)*S + (tj*T + j)] = aOut[hvx_hmx_i8_out_off(i, j)];
        }
    }
}
