/* Near-miss: clears the HMX accumulator INSIDE the K-tile loop (mxclracc every
 * iteration), so only the LAST K-tile survives instead of the chained K=64 sum.
 * Compiles, uses real HMX, but is numerically wrong -> must FAIL the bit-exact
 * gate. Guards against "cleared the accumulator in the wrong place". */
#include "kernel_api.h"
#include "harness_common.h"
#include <hexagon_types.h>

static inline int sx12(int field) {
    int v = field & 0xFFF;
    if (v & 0x800) v -= 0x1000;
    return v;
}

void candidate_kernel(const uint8_t *A, const int8_t *B, int32_t *out, int n, int k_dim) {
    uint8_t          *vAct  = (uint8_t  *)(uintptr_t)(HVX_VTCM_BASE + 0x0000u);
    int8_t           *vWgt  = (int8_t   *)(uintptr_t)(HVX_VTCM_BASE + 0x1000u);
    volatile uint8_t *vBias = (volatile uint8_t *)(uintptr_t)(HVX_VTCM_BASE + 0x2000u);
    uint16_t         *vOut  = (uint16_t *)(uintptr_t)(HVX_VTCM_BASE + 0x3000u);
    static uint8_t  aAct[2048]  HVX_ALIGN;
    static int8_t   aWgt[1024]  HVX_ALIGN;
    static uint16_t aOut[32*32] HVX_ALIGN;
    for (int i = 0; i < 2048; i++) vBias[i] = 0x40;
    const unsigned lim_a = 2047, lim_w = 1023, lim_o = 2047;
    const int T = 32, kt_count = k_dim / T;
    for (int kt = 0; kt < kt_count; kt++) {
        __asm__ volatile("mxclracc\n");        /* BUG: clears every tile */
        for (int i = 0; i < 2048; i++) aAct[i] = 0;
        for (int i = 0; i < n; i++)
            for (int k = 0; k < T; k++) aAct[hvx_hmx_i8_act_off(i, k)] = A[i*k_dim + kt*T + k];
        for (int k = 0; k < T; k++)
            for (int j = 0; j < n; j++) aWgt[hvx_hmx_i8_wgt_off(k, j)] = B[(kt*T + k)*n + j];
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
    for (int i = 0; i < n; i++)
        for (int j = 0; j < n; j++) out[i*n + j] = sx12((int)aOut[hvx_hmx_i8_out_off(i, j)]);
}
