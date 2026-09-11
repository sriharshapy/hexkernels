/* Near-miss: identical HMX tiled flow but packs B TRANSPOSED (B[j*n+k] instead of
 * B[k*n+j]) into the weight tile, so it computes A*B^T. Compiles, uses real HMX,
 * but is numerically wrong -> must FAIL the int8 gate. Guards against credit for
 * merely "uses HMX". */
#include "kernel_api.h"
#include "harness_common.h"

static inline int8_t hmx_i8_narrow(uint16_t field12) {
    int v = (int)(field12 & 0xFFFu);
    if (v & 0x800) v -= 0x1000;
    return (int8_t)v;
}

void candidate_kernel(const uint8_t *A, const int8_t *B, int8_t *out, int n) {
    volatile uint8_t  *vAct  = (volatile uint8_t  *)(uintptr_t)(HVX_VTCM_BASE + 0x0000u);
    volatile int8_t   *vWgt  = (volatile int8_t   *)(uintptr_t)(HVX_VTCM_BASE + 0x1000u);
    volatile uint8_t  *vBias = (volatile uint8_t  *)(uintptr_t)(HVX_VTCM_BASE + 0x2000u);
    volatile uint16_t *vOut  = (volatile uint16_t *)(uintptr_t)(HVX_VTCM_BASE + 0x3000u);

    for (int i = 0; i < 2048; i++) vAct[i]  = 0;
    for (int i = 0; i < 2048; i++) vBias[i] = 0x40;

    for (int i = 0; i < n; i++)
        for (int k = 0; k < n; k++) vAct[hvx_hmx_i8_act_off(i, k)] = A[i*n + k];
    for (int k = 0; k < n; k++)
        for (int j = 0; j < n; j++) vWgt[hvx_hmx_i8_wgt_off(k, j)] = B[j*n + k]; /* transposed */

    unsigned lim_a = 2047, lim_w = 1023, lim_o = 2047;
    __asm__ volatile("mxclracc\n");
    __asm__ volatile("{ activation.ub=mxmem(%0,%1)\n\t weight.b=mxmem(%2,%3) }\n"
                     :: "r"(vAct), "r"(lim_a), "r"(vWgt), "r"(lim_w) : "memory");
    __asm__ volatile("bias=mxmem(%0)\n" :: "r"(vBias) : "memory");
    __asm__ volatile("mxmem(%0,%1):after.uh=acc:2x1\n" :: "r"(vOut), "r"(lim_o) : "memory");
    __asm__ volatile("isync\n\t");

    for (int i = 0; i < n; i++)
        for (int j = 0; j < n; j++) out[i*n + j] = hmx_i8_narrow(vOut[hvx_hmx_i8_out_off(i, j)]);
}
