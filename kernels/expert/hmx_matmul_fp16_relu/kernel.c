/* HMX fp16 32x32 matmul + ReLU (tolerance-correct) — ACCELERATED expert. One
 * HMX matmul in the fp16 crouton layout; ReLU fused as a native HVX Vhf vmax vs
 * zero on the crouton-order output blocks before the plain unpack. Crouton
 * pack/unpack staged via cacheable buffers + bulk 128B copies. */
#include "kernel_api.h"
#include "harness_common.h"
#include <hexagon_types.h>
#include <hexagon_protos.h>
#include <hmx_hexagon_protos.h>

void candidate_kernel(const hvx_hf *A, const hvx_hf *B, hvx_hf *out, int n) {
    hvx_hf *vA = (hvx_hf *)(uintptr_t)(HVX_VTCM_BASE + 0x0000u);
    hvx_hf *vB = (hvx_hf *)(uintptr_t)(HVX_VTCM_BASE + 0x0800u);
    hvx_hf *vO = (hvx_hf *)(uintptr_t)(HVX_VTCM_BASE + 0x1000u);

    static hvx_hf aA[32*32] HVX_ALIGN, aB[32*32] HVX_ALIGN, aO[32*32] HVX_ALIGN;

    for (int r = 0; r < n; r++)
        for (int c = 0; c < n; c++) {
            aA[hvx_crouton_off(r, c)] = A[r*n + c];
            aB[hvx_crouton_off(r, c)] = B[r*n + c];
        }
    { HVX_Vector *s = (HVX_Vector *)aA, *d = (HVX_Vector *)vA;
      for (int b = 0; b < 16; b++) d[b] = s[b]; }
    { HVX_Vector *s = (HVX_Vector *)aB, *d = (HVX_Vector *)vB;
      for (int b = 0; b < 16; b++) d[b] = s[b]; }

    Q6_mxclracc_hf();
    Q6_activation_hf_mxmem_RR((unsigned int)(uintptr_t)vA, 2047);
    Q6_weight_hf_mxmem_RR((unsigned int)(uintptr_t)vB, 2047);
    Q6_mxmem_AR_after_hf(vO, 2047);
    __asm__ volatile("isync\n\t");

    HVX_Vector zeroVec = Q6_Vh_vsplat_R(0);
    { HVX_Vector *s = (HVX_Vector *)vO, *d = (HVX_Vector *)aO;
      for (int b = 0; b < 16; b++) d[b] = Q6_Vhf_vmax_VhfVhf(s[b], zeroVec); }  /* ReLU */
    for (int r = 0; r < n; r++)
        for (int c = 0; c < n; c++) out[r*n + c] = aO[hvx_crouton_off(r, c)];
}
