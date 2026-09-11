/* HMX int8 3x3 conv (VALID) via im2col — the ACCELERATED expert.
 * Lower the conv to a matmul: rows = output positions p (M=P), reduction =
 * receptive field (ci,kh,kw) (K = C_in*KH*KW = 72), cols = output channels (N=C_out).
 * im2col gathers each output position's 3x3xC_in receptive field into a crouton
 * activation row while packing; K=72 is zero-padded up to 3 crouton K-tiles. One
 * mxclracc per output tile, then the 3 K-tile matmul packets accumulate.
 *
 * PERF: the im2col decode is precomputed into flat offset tables so the pack hot loop
 * is division-free: in-index = koff[kg] + poff[p] (koff = ci*IH*IW+kh*IW+kw, poff =
 * oh*IW+ow), and with the k-index ordered kg = ci*KH*KW+kh*KW+kw the weight value is
 * simply W[co*KK + kg]. Crouton pack/unpack is STAGED via cacheable buffers + bulk
 * 128B vector copies (scalar VTCM access ~48 cyc in timing mode would lose). 0x40 requant.
 * Recipe: docs/superpowers/specs/2026-06-12-hmx-int8-layout-re-agenda.md */
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

    for (int i = 0; i < 2048; i++) aAct[i]  = 0;    /* low bytes + pad stay 0 */
    for (int i = 0; i < 1024; i++) aWgt[i]  = 0;
    for (int i = 0; i < 2048; i++) vBias[i] = 0x40;

    /* precompute division-free decode tables */
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
                    out[co*P_OUT + p] = aOut[hvx_hmx_i8_out_off(i, j)];
                }
            }
        }
    }
    (void)n;
}
