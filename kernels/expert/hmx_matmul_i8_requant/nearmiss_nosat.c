/* Near-miss: correct HMX matmul + requant field, but narrows to int8 by
 * TRUNCATION (no saturation). Whenever |r| > 127 the cast wraps modulo 256
 * instead of clamping -> must FAIL bit-exact (the input range makes r exceed the
 * int8 range often). */
#include "kernel_api.h"
#include "harness_common.h"
#include <hexagon_types.h>

void candidate_kernel(const uint8_t *A, const int8_t *B, int8_t *out, int n) {
    uint8_t          *vAct  = (uint8_t  *)(uintptr_t)(HVX_VTCM_BASE + 0x0000u);
    int8_t           *vWgt  = (int8_t   *)(uintptr_t)(HVX_VTCM_BASE + 0x1000u);
    volatile uint8_t *vBias = (volatile uint8_t *)(uintptr_t)(HVX_VTCM_BASE + 0x2000u);
    uint16_t         *vOut  = (uint16_t *)(uintptr_t)(HVX_VTCM_BASE + 0x3000u);

    static uint8_t  aAct[2048]  HVX_ALIGN;
    static int8_t   aWgt[1024]  HVX_ALIGN;
    static uint16_t aOut[32*32] HVX_ALIGN;

    for (int i = 0; i < 2048; i++) aAct[i]  = 0;
    for (int i = 0; i < 2048; i++) vBias[i] = 0x40;

    for (int i = 0; i < n; i++)
        for (int k = 0; k < n; k++) aAct[hvx_hmx_i8_act_off(i, k)] = A[i*n + k];
    for (int k = 0; k < n; k++)
        for (int j = 0; j < n; j++) aWgt[hvx_hmx_i8_wgt_off(k, j)] = B[k*n + j];

    { HVX_Vector *s = (HVX_Vector *)aAct, *d = (HVX_Vector *)vAct;
      for (int b = 0; b < 16; b++) d[b] = s[b]; }
    { HVX_Vector *s = (HVX_Vector *)aWgt, *d = (HVX_Vector *)vWgt;
      for (int b = 0; b < 8; b++) d[b] = s[b]; }

    const unsigned lim_a = 2047, lim_w = 1023, lim_o = 2047;
    __asm__ volatile("mxclracc\n");
    __asm__ volatile("{ activation.ub=mxmem(%0,%1)\n\t weight.b=mxmem(%2,%3) }\n"
                     :: "r"(vAct), "r"(lim_a), "r"(vWgt), "r"(lim_w) : "memory");
    __asm__ volatile("bias=mxmem(%0)\n" :: "r"(vBias) : "memory");
    __asm__ volatile("mxmem(%0,%1):after.uh=acc:2x1\n" :: "r"(vOut), "r"(lim_o) : "memory");
    __asm__ volatile("isync\n\t");

    { HVX_Vector *s = (HVX_Vector *)vOut, *d = (HVX_Vector *)aOut;
      for (int b = 0; b < 16; b++) d[b] = s[b]; }
    for (int i = 0; i < n; i++)
        for (int j = 0; j < n; j++) {
            int v = aOut[hvx_hmx_i8_out_off(i, j)] & 0xFFF;
            if (v & 0x800) v -= 0x1000;
            out[i*n + j] = (int8_t)v;   /* truncating cast -- no saturation */
        }
}
