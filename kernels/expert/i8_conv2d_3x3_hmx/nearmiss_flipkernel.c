/* Near-miss: identical im2col->HMX flow but packs the weight with kh/kw SWAPPED
 * (transposed 3x3 kernel: W[co][ci][kw][kh]). Compiles, uses real HMX, but computes
 * a spatially-transposed convolution -> numerically wrong for a random kernel ->
 * must FAIL the bit-exact gate. Guards against credit for merely "uses HMX". */
#include "kernel_api.h"
#include "harness_common.h"
#include <hexagon_types.h>

void candidate_kernel(const uint8_t *in, const int8_t *W, uint16_t *out, int n) {
    uint8_t          *vAct  = (uint8_t  *)(uintptr_t)(HVX_VTCM_BASE + 0x0000u);
    int8_t           *vWgt  = (int8_t   *)(uintptr_t)(HVX_VTCM_BASE + 0x1000u);
    volatile uint8_t *vBias = (volatile uint8_t *)(uintptr_t)(HVX_VTCM_BASE + 0x2000u);
    uint16_t         *vOut  = (uint16_t *)(uintptr_t)(HVX_VTCM_BASE + 0x3000u);

    static uint8_t  aAct[2048]  HVX_ALIGN;
    static int8_t   aWgt[1024]  HVX_ALIGN;
    static uint16_t aOut[32*32] HVX_ALIGN;

    for (int i = 0; i < 2048; i++) aAct[i]  = 0;
    for (int i = 0; i < 1024; i++) aWgt[i]  = 0;
    for (int i = 0; i < 2048; i++) vBias[i] = 0x40;

    const unsigned lim_a = 2047, lim_w = 1023, lim_o = 2047;
    const int T = 32;
    const int mt = (P_OUT + T - 1) / T, nt = (C_OUT + T - 1) / T, kn = (KK + T - 1) / T;

    for (int ti = 0; ti < mt; ti++) {
        for (int tj = 0; tj < nt; tj++) {
            __asm__ volatile("mxclracc\n");
            for (int kt = 0; kt < kn; kt++) {
                for (int i = 0; i < T; i++) {
                    int p = ti*T + i, oh = p / OW, ow = p % OW;
                    for (int k = 0; k < T; k++) {
                        int kg = kt*T + k; uint8_t v = 0;
                        if (p < P_OUT && kg < KK) {
                            int ci = kg / (KH*KW), r = kg % (KH*KW);
                            int kh = r / KW, kw = r % KW;
                            v = in[ci*IH*IW + (oh*STRIDE + kh)*IW + (ow*STRIDE + kw)];
                        }
                        aAct[hvx_hmx_i8_act_off(i, k)] = v;
                    }
                }
                for (int k = 0; k < T; k++) {
                    int kg = kt*T + k;
                    for (int j = 0; j < T; j++) {
                        int co = tj*T + j; int8_t v = 0;
                        if (co < C_OUT && kg < KK) {
                            int ci = kg / (KH*KW), rr = kg % (KH*KW);
                            int kh = rr / KW, kw = rr % KW;
                            v = W[((co*C_IN + ci)*KH + kw)*KW + kh]; /* kh/kw SWAPPED */
                        }
                        aWgt[hvx_hmx_i8_wgt_off(k, j)] = v;
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
                int p = ti*T + i; if (p >= P_OUT) continue;
                for (int j = 0; j < T; j++) {
                    int co = tj*T + j; if (co >= C_OUT) continue;
                    out[co*P_OUT + p] = aOut[hvx_hmx_i8_out_off(i, j)];
                }
            }
        }
    }
    (void)n;
}
