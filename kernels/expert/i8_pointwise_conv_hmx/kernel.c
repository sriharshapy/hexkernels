/* HMX int8 1x1 (pointwise) conv — the ACCELERATED expert.
 * 1x1 conv over channels is a matmul: rows = spatial positions p (M=P), reduction
 * = input channels ci (K=C_in), cols = output channels co (N=C_out). We tile into
 * 32x32 crouton output tiles; each accumulates over C_in/32 K-tiles. The
 * conv-specific work vs a raw matmul is the NCHW<->position-major gather during the
 * activation/weight pack (A[p][ci]=in[ci*P+p], B[ci][co]=W[co*C_in+ci]).
 * Crouton pack/unpack is STAGED through cacheable buffers and moved to/from VTCM in
 * bulk 128B vector copies (scalar VTCM access costs ~48 cyc in timing mode and would
 * lose to the HVX vrmpy baseline). The 0x40-config store applies the requant.
 * Recipe: docs/superpowers/specs/2026-06-12-hmx-int8-layout-re-agenda.md */
#include "kernel_api.h"
#include "harness_common.h"
#include <hexagon_types.h>

void candidate_kernel(const uint8_t *in, const int8_t *W, uint16_t *out, int n) {
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
    const int T = 32;
    const int mt = CONV_P    / T;   /* row (position) tiles */
    const int nt = CONV_COUT / T;   /* col (out-channel) tiles */
    const int kn = CONV_CIN  / T;   /* K (in-channel) tiles */

    for (int ti = 0; ti < mt; ti++) {
        for (int tj = 0; tj < nt; tj++) {
            __asm__ volatile("mxclracc\n");
            for (int kt = 0; kt < kn; kt++) {
                for (int i = 0; i < T; i++)
                    for (int k = 0; k < T; k++)
                        aAct[hvx_hmx_i8_act_off(i, k)] = in[(kt*T + k)*CONV_P + (ti*T + i)];
                for (int k = 0; k < T; k++)
                    for (int j = 0; j < T; j++)
                        aWgt[hvx_hmx_i8_wgt_off(k, j)] = W[(tj*T + j)*CONV_CIN + (kt*T + k)];
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
                    out[(tj*T + j)*CONV_P + (ti*T + i)] = aOut[hvx_hmx_i8_out_off(i, j)];
        }
    }
    (void)n;
}
