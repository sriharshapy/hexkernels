/* Near-miss: packs the weight WITHOUT transposing (uses W[ci][co] directly), so
 * it contracts the wrong axis and computes In*W instead of In*W^T. Compiles,
 * uses real HMX, but is numerically wrong -> must FAIL the tolerance gate. */
#include "kernel_api.h"
#include "harness_common.h"
#include <hexagon_types.h>
#include <hmx_hexagon_protos.h>

void candidate_kernel(const hvx_hf *In, const hvx_hf *W, hvx_hf *out, int n) {
    hvx_hf *vA = (hvx_hf *)(uintptr_t)(HVX_VTCM_BASE + 0x0000u);
    hvx_hf *vB = (hvx_hf *)(uintptr_t)(HVX_VTCM_BASE + 0x0800u);
    hvx_hf *vO = (hvx_hf *)(uintptr_t)(HVX_VTCM_BASE + 0x1000u);
    static hvx_hf aA[32*32] HVX_ALIGN, aB[32*32] HVX_ALIGN, aO[32*32] HVX_ALIGN;
    for (int p = 0; p < n; p++)
        for (int ci = 0; ci < n; ci++) aA[hvx_crouton_off(p, ci)] = In[p*n + ci];
    for (int ci = 0; ci < n; ci++)
        for (int co = 0; co < n; co++) aB[hvx_crouton_off(ci, co)] = W[ci*n + co]; /* not transposed */
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
    for (int p = 0; p < n; p++)
        for (int co = 0; co < n; co++) out[p*n + co] = aO[hvx_crouton_off(p, co)];
}
