/* HMX fp16 32x32 matmul + per-column bias (tolerance-correct) — ACCELERATED
 * expert. One HMX matmul in the fp16 crouton layout; the per-column bias is
 * added DIRECTLY on the crouton-order output blocks (16 vector qf16 adds) by
 * pre-arranging bias into crouton column order (each column duplicated for the
 * two interleaved rows), so no extra un-crouton pass is needed -- then the plain
 * un-crouton store. Crouton pack/unpack staged via cacheable buffers + bulk copies. */
#include "kernel_api.h"
#include "harness_common.h"
#include <hexagon_types.h>
#include <hexagon_protos.h>
#include <hmx_hexagon_protos.h>

void candidate_kernel(const hvx_hf *A, const hvx_hf *B, const hvx_hf *bias,
                       hvx_hf *out, int n) {
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

    /* crouton-order bias: within a crouton vector, hf position p holds column p/2
     * (the two interleaved rows share a column), so bc[p]=bias[p/2] and the SAME
     * bc vector applies to all 16 output blocks. */
    static hvx_hf bc[64] HVX_ALIGN;
    for (int j = 0; j < n; j++) { bc[2*j] = bias[j]; bc[2*j+1] = bias[j]; }
    HVX_Vector vbc = *(const HVX_UVector *)bc;
    { HVX_Vector *s = (HVX_Vector *)vO, *d = (HVX_Vector *)aO;
      for (int b = 0; b < 16; b++)
          d[b] = Q6_Vhf_equals_Vqf16(Q6_Vqf16_vadd_VhfVhf(s[b], vbc)); }  /* + bias */
    for (int r = 0; r < n; r++)
        for (int c = 0; c < n; c++) out[r*n + c] = aO[hvx_crouton_off(r, c)];
}
