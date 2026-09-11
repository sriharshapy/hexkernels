/* HMX int8 3x3 conv (VALID) + per-channel bias + ReLU via im2col — near-miss: bias-add but ReLU OMITTED (negatives not clamped) -> INCORRECT.
 * Identical im2col->HMX lowering as i8_conv2d_3x3_hmx (M=P, K=72 zero-padded to 3
 * crouton K-tiles, N=C_out), but the HVX unpack FUSES the epilogue: sign-extend the
 * HMX 12-bit requant field, add the per-output-channel int32 bias, ReLU-clamp to >=0.
 * Division-free pack via precomputed koff/poff tables; weight value = W[co*KK + kg].
 * Crouton pack/unpack staged via cacheable buffers + bulk 128B vector copies.
 * Recipe: docs/superpowers/specs/2026-06-12-hmx-int8-layout-re-agenda.md */
#include "kernel_api.h"
#include "harness_common.h"
#include <hexagon_types.h>

static inline int sx12(int field) {
    int v = field & 0xFFF;
    if (v & 0x800) v -= 0x1000;
    return v;
}

void candidate_kernel(const uint8_t *in, const int8_t *W, const int32_t *bias,
                      int32_t *out, int n) {
    uint8_t          *vAct  = (uint8_t  *)(uintptr_t)(HVX_VTCM_BASE + 0x0000u);
    int8_t           *vWgt  = (int8_t   *)(uintptr_t)(HVX_VTCM_BASE + 0x1000u);
    volatile uint8_t *vBias = (volatile uint8_t *)(uintptr_t)(HVX_VTCM_BASE + 0x2000u);
    uint16_t         *vOut  = (uint16_t *)(uintptr_t)(HVX_VTCM_BASE + 0x3000u);

    static uint8_t  aAct[2048]  HVX_ALIGN;
    static int8_t   aWgt[1024]  HVX_ALIGN;
    static uint16_t aOut[32*32] HVX_ALIGN;

    for (int i = 0; i < 2048; i++) aAct[i]  = 0;
    for (int i = 0; i < 1024; i++) aWgt[i]  = 0;
    for (int i = 0; i < 2048; i++) vBias[i] = 0x40;   /* HMX requant config: scale 17/16 */

    int koff[KK], poff[P_OUT];
    for (int k = 0; k < KK; k++) {
        int ci = k / (KH*KW), r = k % (KH*KW), kh = r / KW, kw = r % KW;
        koff[k] = ci*IH*IW + kh*IW + kw;
    }
    for (int p = 0; p < P_OUT; p++) poff[p] = (p / OW)*IW + (p % OW);

    const unsigned lim_a = 2047, lim_w = 1023, lim_o = 2047;
    const int T = 32;
    const int mt = (P_OUT + T - 1) / T;
    const int nt = (C_OUT + T - 1) / T;
    const int kn = (KK   + T - 1) / T;

    for (int ti = 0; ti < mt; ti++) {
        for (int tj = 0; tj < nt; tj++) {
            __asm__ volatile("mxclracc\n");
            for (int kt = 0; kt < kn; kt++) {
                for (int i = 0; i < T; i++) {
                    int p = ti*T + i;
                    if (p >= P_OUT) continue;
                    int po = poff[p];
                    for (int k = 0; k < T; k++) {
                        int kg = kt*T + k;
                        aAct[hvx_hmx_i8_act_off(i, k)] = (kg < KK) ? in[koff[kg] + po] : 0;
                    }
                }
                for (int k = 0; k < T; k++) {
                    int kg = kt*T + k;
                    for (int j = 0; j < T; j++) {
                        int co = tj*T + j;
                        aWgt[hvx_hmx_i8_wgt_off(k, j)] =
                            (co < C_OUT && kg < KK) ? W[co*KK + kg] : 0;
                    }
                }
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
            for (int i = 0; i < T; i++) {
                int p = ti*T + i;
                if (p >= P_OUT) continue;
                for (int j = 0; j < T; j++) {
                    int co = tj*T + j;
                    if (co >= C_OUT) continue;
                    int r = sx12((int)aOut[hvx_hmx_i8_out_off(i, j)]);
                    int biased = r + bias[co];
                    out[co*P_OUT + p] = biased; /* near-miss: ReLU omitted */
                }
            }
        }
    }
    (void)n;
}
