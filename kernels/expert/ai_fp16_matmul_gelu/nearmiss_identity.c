/* Near-miss: applies no activation — returns raw matmul result.
 * Compiles but fails because the reference applies GELU. */
#include "kernel_api.h"
#include "harness_common.h"
#include <hexagon_types.h>
#include <hmx_hexagon_protos.h>

void candidate_kernel(const hvx_hf *A, const hvx_hf *B, hvx_hf *out, int n) {
    hvx_hf *vA = (hvx_hf *)(uintptr_t)(HVX_VTCM_BASE + 0x0000u);
    hvx_hf *vB = (hvx_hf *)(uintptr_t)(HVX_VTCM_BASE + 0x0800u);
    hvx_hf *vO = (hvx_hf *)(uintptr_t)(HVX_VTCM_BASE + 0x1000u);
    for (int r = 0; r < n; r++)
        for (int c = 0; c < n; c++) {
            vA[hvx_crouton_off(r, c)] = A[r*n+c];
            vB[hvx_crouton_off(r, c)] = B[r*n+c];
        }
    Q6_mxclracc_hf();
    Q6_activation_hf_mxmem_RR((unsigned int)(uintptr_t)vA, 2047);
    Q6_weight_hf_mxmem_RR((unsigned int)(uintptr_t)vB, 2047);
    Q6_mxmem_AR_after_hf(vO, 2047);
    __asm__ volatile("isync\n\t");
    for (int r = 0; r < n; r++)
        for (int c = 0; c < n; c++)
            out[r*n+c] = vO[hvx_crouton_off(r, c)];  /* no activation */
}
