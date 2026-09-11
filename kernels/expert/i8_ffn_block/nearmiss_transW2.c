/* Near-miss: identical HMX x2 + VTCM + HVX FFN flow, but the SECOND matmul packs
 * the W2 tile TRANSPOSED (W2[(tj*T+j)*Dff + (kt*T+k)] instead of
 * W2[(kt*T+k)*D + (tj*T+j)]), i.e. it computes H*W2^T. Compiles, genuinely uses
 * HMX + VTCM + HVX and applies the ReLU activation correctly, but the down-
 * projection is numerically wrong -> MUST FAIL the bit-exact gate. Guards against
 * credit for merely "composes the mechanisms + gets matmul1 right". */
#include "kernel_api.h"
#include "harness_common.h"
#include <hexagon_types.h>

#define ACT_BYTES 2048u
#define WGT_BYTES 1024u

void candidate_kernel(const uint8_t *X, const int8_t *W1, const int32_t *b1,
                      const int8_t *W2, const int32_t *b2,
                      int8_t *out, int S, int D, int Dff) {
    const uint32_t vAct  = HVX_VTCM_BASE + 0x0000u;
    const uint32_t vWgt  = HVX_VTCM_BASE + 0x0800u;
    const uint32_t vBias = HVX_VTCM_BASE + 0x1000u;
    const uint32_t vOut  = HVX_VTCM_BASE + 0x1800u;
    const uint32_t vH    = HVX_VTCM_BASE + 0x2000u;

    static uint8_t  aAct[ACT_BYTES]  HVX_ALIGN;
    static int8_t   aWgt[WGT_BYTES]  HVX_ALIGN;
    static uint16_t aOut[32*32]      HVX_ALIGN;
    static uint8_t  aBias[ACT_BYTES] HVX_ALIGN;
    static uint8_t  aH [256*256]     HVX_ALIGN;
    static uint8_t  aH2[256*256]     HVX_ALIGN;

    for (int i = 0; i < (int)ACT_BYTES; i++) aAct[i]  = 0;
    for (int i = 0; i < (int)ACT_BYTES; i++) aBias[i] = 0x40;
    { HVX_Vector *sb = (HVX_Vector *)aBias, *db = (HVX_Vector *)(uintptr_t)vBias;
      for (int b = 0; b < 16; b++) db[b] = sb[b]; }

    const unsigned lim_a = 2047, lim_w = 1023, lim_o = 2047;
    const int T = 32;
    const int ntS = S / T, ntDff = Dff / T, ntD = D / T;

    for (int ti = 0; ti < ntS; ti++) {
        for (int tj = 0; tj < ntDff; tj++) {
            __asm__ volatile("mxclracc\n");
            for (int kt = 0; kt < ntD; kt++) {
                for (int i = 0; i < T; i++)
                    for (int k = 0; k < T; k++)
                        aAct[hvx_hmx_i8_act_off(i, k)] = X[(ti*T + i)*D + (kt*T + k)];
                for (int k = 0; k < T; k++)
                    for (int j = 0; j < T; j++)
                        aWgt[hvx_hmx_i8_wgt_off(k, j)] = W1[(kt*T + k)*Dff + (tj*T + j)];
                { HVX_Vector *s = (HVX_Vector *)aAct, *d = (HVX_Vector *)(uintptr_t)vAct;
                  for (int b = 0; b < 16; b++) d[b] = s[b]; }
                { HVX_Vector *s = (HVX_Vector *)aWgt, *d = (HVX_Vector *)(uintptr_t)vWgt;
                  for (int b = 0; b < 8; b++) d[b] = s[b]; }
                __asm__ volatile("{ activation.ub=mxmem(%0,%1)\n\t weight.b=mxmem(%2,%3) }\n"
                                 :: "r"(vAct), "r"(lim_a), "r"(vWgt), "r"(lim_w) : "memory");
            }
            __asm__ volatile("bias=mxmem(%0)\n" :: "r"(vBias) : "memory");
            __asm__ volatile("mxmem(%0,%1):after.uh=acc:2x1\n" :: "r"(vOut), "r"(lim_o) : "memory");
            __asm__ volatile("isync\n\t");
            { HVX_Vector *s = (HVX_Vector *)(uintptr_t)vOut, *d = (HVX_Vector *)aOut;
              for (int b = 0; b < 16; b++) d[b] = s[b]; }
            for (int i = 0; i < T; i++)
                for (int j = 0; j < T; j++) {
                    int r1 = ffn_sx12((int)aOut[hvx_hmx_i8_out_off(i, j)]);
                    int jj = tj*T + j;
                    aH[(ti*T + i)*Dff + jj] = (uint8_t)ffn_relu_requant(r1 + b1[jj]);
                }
        }
    }

    const int nvH = (S*Dff) / 128;
    { HVX_Vector *s = (HVX_Vector *)aH, *d = (HVX_Vector *)(uintptr_t)vH;
      for (int b = 0; b < nvH; b++) d[b] = s[b]; }
    { HVX_Vector *s = (HVX_Vector *)(uintptr_t)vH, *d = (HVX_Vector *)aH2;
      for (int b = 0; b < nvH; b++) d[b] = s[b]; }

    for (int ti = 0; ti < ntS; ti++) {
        for (int tj = 0; tj < ntD; tj++) {
            __asm__ volatile("mxclracc\n");
            for (int kt = 0; kt < ntDff; kt++) {
                for (int i = 0; i < T; i++)
                    for (int k = 0; k < T; k++)
                        aAct[hvx_hmx_i8_act_off(i, k)] = aH2[(ti*T + i)*Dff + (kt*T + k)];
                for (int k = 0; k < T; k++)
                    for (int j = 0; j < T; j++)
                        aWgt[hvx_hmx_i8_wgt_off(k, j)] = W2[(tj*T + j)*Dff + (kt*T + k)]; /* transposed */
                { HVX_Vector *s = (HVX_Vector *)aAct, *d = (HVX_Vector *)(uintptr_t)vAct;
                  for (int b = 0; b < 16; b++) d[b] = s[b]; }
                { HVX_Vector *s = (HVX_Vector *)aWgt, *d = (HVX_Vector *)(uintptr_t)vWgt;
                  for (int b = 0; b < 8; b++) d[b] = s[b]; }
                __asm__ volatile("{ activation.ub=mxmem(%0,%1)\n\t weight.b=mxmem(%2,%3) }\n"
                                 :: "r"(vAct), "r"(lim_a), "r"(vWgt), "r"(lim_w) : "memory");
            }
            __asm__ volatile("bias=mxmem(%0)\n" :: "r"(vBias) : "memory");
            __asm__ volatile("mxmem(%0,%1):after.uh=acc:2x1\n" :: "r"(vOut), "r"(lim_o) : "memory");
            __asm__ volatile("isync\n\t");
            { HVX_Vector *s = (HVX_Vector *)(uintptr_t)vOut, *d = (HVX_Vector *)aOut;
              for (int b = 0; b < 16; b++) d[b] = s[b]; }
            for (int i = 0; i < T; i++)
                for (int j = 0; j < T; j++) {
                    int r2 = ffn_sx12((int)aOut[hvx_hmx_i8_out_off(i, j)]);
                    int jj = tj*T + j;
                    out[(ti*T + i)*D + jj] = ffn_sat_i8((r2 + b2[jj]) >> FFN_SH2);
                }
        }
    }
}
