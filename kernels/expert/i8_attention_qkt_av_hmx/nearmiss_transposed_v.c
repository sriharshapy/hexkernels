/* Near-miss: identical fused HMX QK^T-then-AV flow, but stage 2 packs V
 * TRANSPOSED (weight[k][j] = V[j][k] instead of V[k][j]), so it computes
 * Sc . V^T instead of Sc . V. Compiles, uses real HMX for both stages, stage 1
 * (scores) is correct, but the final AV matmul is numerically wrong for a
 * random V -> must FAIL the bit-exact gate. */
#include "kernel_api.h"
#include "harness_common.h"
#include <hexagon_types.h>

static inline int sx12(int field) {
    int v = field & 0xFFF;
    if (v & 0x800) v -= 0x1000;
    return v;
}

void candidate_kernel(const uint8_t *Q, const int8_t *K, const int8_t *V,
                       int32_t *out, int S, int D) {
    uint8_t          *vAct  = (uint8_t  *)(uintptr_t)(HVX_VTCM_BASE + 0x0000u);
    int8_t           *vWgt  = (int8_t   *)(uintptr_t)(HVX_VTCM_BASE + 0x1000u);
    volatile uint8_t *vBias = (volatile uint8_t *)(uintptr_t)(HVX_VTCM_BASE + 0x2000u);
    uint16_t         *vOut  = (uint16_t *)(uintptr_t)(HVX_VTCM_BASE + 0x3000u);

    static uint8_t  aAct[2048]  HVX_ALIGN;
    static int8_t   aWgt[1024]  HVX_ALIGN;
    static uint16_t aOut[32*32] HVX_ALIGN;
    static uint8_t  Sc[32*32]   HVX_ALIGN;

    const unsigned lim_a = 2047, lim_w = 1023, lim_o = 2047;
    const int T = 32;

    for (int i = 0; i < 2048; i++) aAct[i]  = 0;
    for (int i = 0; i < 2048; i++) vBias[i] = 0x40;

    /* Stage 1: correct QK^T */
    for (int i = 0; i < T; i++)
        for (int d = 0; d < T; d++)
            aAct[hvx_hmx_i8_act_off(i, d)] = Q[i*D + d];
    for (int d = 0; d < T; d++)
        for (int j = 0; j < T; j++)
            aWgt[hvx_hmx_i8_wgt_off(d, j)] = K[j*D + d];
    { HVX_Vector *s = (HVX_Vector *)aAct, *d = (HVX_Vector *)vAct;
      for (int b = 0; b < 16; b++) d[b] = s[b]; }
    { HVX_Vector *s = (HVX_Vector *)aWgt, *d = (HVX_Vector *)vWgt;
      for (int b = 0; b < 8; b++) d[b] = s[b]; }
    __asm__ volatile("mxclracc\n");
    __asm__ volatile("{ activation.ub=mxmem(%0,%1)\n\t weight.b=mxmem(%2,%3) }\n"
                     :: "r"(vAct), "r"(lim_a), "r"(vWgt), "r"(lim_w) : "memory");
    __asm__ volatile("bias=mxmem(%0)\n" :: "r"(vBias) : "memory");
    __asm__ volatile("mxmem(%0,%1):after.uh=acc:2x1\n" :: "r"(vOut), "r"(lim_o) : "memory");
    __asm__ volatile("isync\n\t");
    { HVX_Vector *s = (HVX_Vector *)vOut, *d = (HVX_Vector *)aOut;
      for (int b = 0; b < 16; b++) d[b] = s[b]; }
    for (int i = 0; i < T; i++)
        for (int j = 0; j < T; j++) {
            int v = sx12((int)aOut[hvx_hmx_i8_out_off(i, j)]);
            Sc[i*T + j] = (uint8_t)v;
        }

    /* Stage 2: BUG -- V packed transposed */
    for (int i = 0; i < T; i++)
        for (int k = 0; k < T; k++)
            aAct[hvx_hmx_i8_act_off(i, k)] = Sc[i*T + k];
    for (int k = 0; k < T; k++)
        for (int j = 0; j < T; j++)
            aWgt[hvx_hmx_i8_wgt_off(k, j)] = V[j*D + k];   /* transposed */
    { HVX_Vector *s = (HVX_Vector *)aAct, *d = (HVX_Vector *)vAct;
      for (int b = 0; b < 16; b++) d[b] = s[b]; }
    { HVX_Vector *s = (HVX_Vector *)aWgt, *d = (HVX_Vector *)vWgt;
      for (int b = 0; b < 8; b++) d[b] = s[b]; }
    __asm__ volatile("mxclracc\n");
    __asm__ volatile("{ activation.ub=mxmem(%0,%1)\n\t weight.b=mxmem(%2,%3) }\n"
                     :: "r"(vAct), "r"(lim_a), "r"(vWgt), "r"(lim_w) : "memory");
    __asm__ volatile("bias=mxmem(%0)\n" :: "r"(vBias) : "memory");
    __asm__ volatile("mxmem(%0,%1):after.uh=acc:2x1\n" :: "r"(vOut), "r"(lim_o) : "memory");
    __asm__ volatile("isync\n\t");
    { HVX_Vector *s = (HVX_Vector *)vOut, *d = (HVX_Vector *)aOut;
      for (int b = 0; b < 16; b++) d[b] = s[b]; }
    for (int i = 0; i < S; i++)
        for (int d = 0; d < D; d++)
            out[i*D + d] = sx12((int)aOut[hvx_hmx_i8_out_off(i, d)]);
}
