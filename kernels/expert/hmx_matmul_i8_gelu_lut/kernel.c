/* HMX int8 32x32 matmul + GELU-via-LUT epilogue (bit-exact) — ACCELERATED
 * expert. One HMX matmul; the 0x40-config store requantizes to a small signed
 * field, then the HVX epilogue sign-extends it and applies GELU by indexing the
 * provided 256-entry LUT (out = lut[r+128]). Crouton pack/unpack staged via
 * cacheable buffers + bulk 128B copies. */
#include "kernel_api.h"
#include "harness_common.h"
#include <hexagon_types.h>

static inline int sx12(int field) {
    int v = field & 0xFFF;
    if (v & 0x800) v -= 0x1000;
    return v;
}

void candidate_kernel(const uint8_t *A, const int8_t *B, const int32_t *lut,
                       int32_t *out, int n) {
    uint8_t          *vAct  = (uint8_t  *)(uintptr_t)(HVX_VTCM_BASE + 0x0000u);
    int8_t           *vWgt  = (int8_t   *)(uintptr_t)(HVX_VTCM_BASE + 0x1000u);
    volatile uint8_t *vBias = (volatile uint8_t *)(uintptr_t)(HVX_VTCM_BASE + 0x2000u);
    uint16_t         *vOut  = (uint16_t *)(uintptr_t)(HVX_VTCM_BASE + 0x3000u);

    static uint8_t  aAct[2048]  HVX_ALIGN;
    static int8_t   aWgt[1024]  HVX_ALIGN;
    static uint16_t aOut[32*32] HVX_ALIGN;

    for (int i = 0; i < 2048; i++) aAct[i]  = 0;
    for (int i = 0; i < 2048; i++) vBias[i] = 0x40;
    const unsigned lim_a = 2047, lim_w = 1023, lim_o = 2047;

    __asm__ volatile("mxclracc\n");
    for (int i = 0; i < n; i++)
        for (int k = 0; k < n; k++) aAct[hvx_hmx_i8_act_off(i, k)] = A[i*n + k];
    for (int k = 0; k < n; k++)
        for (int j = 0; j < n; j++) aWgt[hvx_hmx_i8_wgt_off(k, j)] = B[k*n + j];
    { HVX_Vector *s = (HVX_Vector *)aAct, *d = (HVX_Vector *)vAct;
      for (int b = 0; b < 16; b++) d[b] = s[b]; }
    { HVX_Vector *s = (HVX_Vector *)aWgt, *d = (HVX_Vector *)vWgt;
      for (int b = 0; b < 8; b++) d[b] = s[b]; }
    __asm__ volatile("{ activation.ub=mxmem(%0,%1)\n\t weight.b=mxmem(%2,%3) }\n"
                     :: "r"(vAct), "r"(lim_a), "r"(vWgt), "r"(lim_w) : "memory");
    __asm__ volatile("bias=mxmem(%0)\n" :: "r"(vBias) : "memory");
    __asm__ volatile("mxmem(%0,%1):after.uh=acc:2x1\n" :: "r"(vOut), "r"(lim_o) : "memory");
    __asm__ volatile("isync\n\t");
    { HVX_Vector *s = (HVX_Vector *)vOut, *d = (HVX_Vector *)aOut;
      for (int b = 0; b < 16; b++) d[b] = s[b]; }
    for (int i = 0; i < n; i++)
        for (int j = 0; j < n; j++) {
            int r = sx12((int)aOut[hvx_hmx_i8_out_off(i, j)]);
            out[i*n + j] = lut[r + 128];          /* GELU via LUT */
        }
}
