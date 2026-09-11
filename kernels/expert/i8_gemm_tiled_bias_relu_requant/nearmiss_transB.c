/* Near-miss: identical HMX+DMA+VTCM tiled flow, but packs the weight tile
 * TRANSPOSED (B[(tj*T+j)*N + (kt*T+k)] instead of B[(kt*T+k)*N + (tj*T+j)]), so
 * it computes A*B^T. Compiles, genuinely uses HMX + uDMA + VTCM double-buffer,
 * but is numerically wrong -> MUST FAIL the bit-exact gate. Guards against credit
 * for merely "composes the mechanisms". */
#include "kernel_api.h"
#include "harness_common.h"
#include <hexagon_types.h>
#include <hexagon_protos.h>

typedef struct __attribute__((aligned(32))) { uint32_t next, ctrl, src, dst; } desc_t;
static desc_t d_a, d_w;

static inline int sx12(int field) {
    int v = field & 0xFFF;
    if (v & 0x800) v -= 0x1000;
    return v;
}
static inline uint8_t saturate_u8(int v) {
    if (v > 255) v = 255;
    if (v <   0) v = 0;
    return (uint8_t)v;
}

#define ACT_BYTES 2048u
#define WGT_BYTES 1024u

void candidate_kernel(const uint8_t *A, const int8_t *B, const int32_t *bias,
                      uint8_t *out, int M, int N, int K) {
    const uint32_t vAct0 = HVX_VTCM_BASE + 0x0000u;
    const uint32_t vAct1 = HVX_VTCM_BASE + 0x0800u;
    const uint32_t vWgt0 = HVX_VTCM_BASE + 0x1000u;
    const uint32_t vWgt1 = HVX_VTCM_BASE + 0x1400u;
    const uint32_t vBias = HVX_VTCM_BASE + 0x2000u;
    const uint32_t vOut  = HVX_VTCM_BASE + 0x3000u;

    static uint8_t aAct[2][ACT_BYTES] HVX_ALIGN;
    static int8_t  aWgt[2][WGT_BYTES] HVX_ALIGN;
    static uint16_t aOut[32*32]       HVX_ALIGN;
    static uint8_t  aBias[ACT_BYTES]  HVX_ALIGN;

    for (int p = 0; p < 2; p++) for (int i = 0; i < (int)ACT_BYTES; i++) aAct[p][i] = 0;
    for (int i = 0; i < (int)ACT_BYTES; i++) aBias[i] = 0x40;
    { HVX_Vector *sb = (HVX_Vector *)aBias, *db = (HVX_Vector *)(uintptr_t)vBias;
      for (int b = 0; b < 16; b++) db[b] = sb[b]; }

    const unsigned lim_a = 2047, lim_w = 1023, lim_o = 2047;
    const int T = 32;
    const int ntM = M / T, ntN = N / T, nkt = K / T;

    for (int ti = 0; ti < ntM; ti++) {
        for (int tj = 0; tj < ntN; tj++) {
            for (int i = 0; i < T; i++)
                for (int k = 0; k < T; k++)
                    aAct[0][hvx_hmx_i8_act_off(i, k)] = A[(ti*T + i)*K + (0*T + k)];
            for (int k = 0; k < T; k++)
                for (int j = 0; j < T; j++)
                    aWgt[0][hvx_hmx_i8_wgt_off(k, j)] = B[(tj*T + j)*N + (0*T + k)]; /* transposed */
            d_a.next = (uint32_t)(uintptr_t)&d_w; d_a.ctrl = ACT_BYTES;
            d_a.src = (uint32_t)(uintptr_t)aAct[0]; d_a.dst = vAct0;
            d_w.next = 0; d_w.ctrl = WGT_BYTES;
            d_w.src = (uint32_t)(uintptr_t)aWgt[0]; d_w.dst = vWgt0;
            Q6_dmstart_A(&d_a); Q6_R_dmwait();

            __asm__ volatile("mxclracc\n");

            for (int kt = 0; kt < nkt; kt++) {
                int cur = kt & 1;
                uint32_t vAct_cur = cur ? vAct1 : vAct0;
                uint32_t vWgt_cur = cur ? vWgt1 : vWgt0;
                if (kt + 1 < nkt) {
                    int nb = (kt + 1) & 1;
                    for (int i = 0; i < T; i++)
                        for (int k = 0; k < T; k++)
                            aAct[nb][hvx_hmx_i8_act_off(i, k)] = A[(ti*T + i)*K + ((kt+1)*T + k)];
                    for (int k = 0; k < T; k++)
                        for (int j = 0; j < T; j++)
                            aWgt[nb][hvx_hmx_i8_wgt_off(k, j)] = B[(tj*T + j)*N + ((kt+1)*T + k)]; /* transposed */
                    d_a.next = (uint32_t)(uintptr_t)&d_w; d_a.ctrl = ACT_BYTES;
                    d_a.src = (uint32_t)(uintptr_t)aAct[nb]; d_a.dst = nb ? vAct1 : vAct0;
                    d_w.next = 0; d_w.ctrl = WGT_BYTES;
                    d_w.src = (uint32_t)(uintptr_t)aWgt[nb]; d_w.dst = nb ? vWgt1 : vWgt0;
                    Q6_dmstart_A(&d_a);
                }
                __asm__ volatile("{ activation.ub=mxmem(%0,%1)\n\t weight.b=mxmem(%2,%3) }\n"
                                 :: "r"(vAct_cur), "r"(lim_a), "r"(vWgt_cur), "r"(lim_w) : "memory");
                if (kt + 1 < nkt) Q6_R_dmwait();
            }

            __asm__ volatile("bias=mxmem(%0)\n" :: "r"(vBias) : "memory");
            __asm__ volatile("mxmem(%0,%1):after.uh=acc:2x1\n" :: "r"(vOut), "r"(lim_o) : "memory");
            __asm__ volatile("isync\n\t");
            { HVX_Vector *s = (HVX_Vector *)(uintptr_t)vOut, *d = (HVX_Vector *)aOut;
              for (int b = 0; b < 16; b++) d[b] = s[b]; }
            for (int i = 0; i < T; i++)
                for (int j = 0; j < T; j++) {
                    int r      = sx12((int)aOut[hvx_hmx_i8_out_off(i, j)]);
                    int jj     = tj*T + j;
                    int biased = r + bias[jj];
                    int relu   = biased > 0 ? biased : 0;
                    out[(ti*T + i)*N + jj] = saturate_u8(relu);
                }
        }
    }
}
