/* HMX fp16 32x32 fused tile MAC (tolerance-correct) — the ACCELERATED expert.
 * Runs one HMX matmul A*B in the fp16 crouton layout, then ADDS the product into
 * the pre-loaded C0 accumulator during the un-crouton unpack (the MAC fusion).
 * Scalar VTCM accesses cost ~48 cyc each in timing mode, so crouton pack/unpack
 * is staged through cacheable buffers moved to/from VTCM in bulk 128B copies. */
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

    { HVX_Vector *s = (HVX_Vector *)vO, *d = (HVX_Vector *)aO;
      for (int b = 0; b < 16; b++) d[b] = s[b]; }
    /* cheap plain un-crouton to row-major, then vectorized MAC add into C0 */
    static hvx_hf rm[32*32 + 64] HVX_ALIGN;
    for (int r = 0; r < n; r++)
        for (int c = 0; c < n; c++) rm[r*n + c] = aO[hvx_crouton_off(r, c)];
    for (int r = 0; r < n; r++) {
        HVX_Vector prod = *(const HVX_UVector *)(rm + r*n);
        HVX_Vector c0   = *(const HVX_UVector *)(out + r*n);
        HVX_Vector out_hf = Q6_Vhf_equals_Vqf16(Q6_Vqf16_vadd_VhfVhf(prod, c0));
        const hvx_hf *rp = (const hvx_hf *)&out_hf;
        for (int j = 0; j < n; j++) out[r*n + j] = rp[j];   /* store 32 lanes */
    }
}
