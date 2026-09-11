/* sol_03: HVX-vectorized ReLU backward using qfloat comparison mask.
 * Strategy: compare x > 0 via Q6_Vqf32_vsub to get sign, use vand/vand to gate dy.
 * 32 fp32 lanes per 128-byte HVX vector.
 * Semantics: dx[i] = (x[i] > 0.0f) ? dy[i] : 0.0f */
#include "kernel_api.h"
#include <hexagon_types.h>
#include <hexagon_protos.h>

void candidate_kernel(const float *x, const float *dy, float *dx, int n) {
    /* zero vector for comparison */
    HVX_Vector vzero_sf = Q6_V_vzero();

    int nvec = n / 32;
    int rem  = n - nvec * 32;

    for (int k = 0; k < nvec; k++) {
        HVX_Vector vx  = *(const HVX_Vector *)(x  + k * 32);
        HVX_Vector vdy = *(const HVX_Vector *)(dy + k * 32);

        /* Compare x > 0: use Q6_Q_vcmp_gt_VsfVsf (returns predicate vector) */
        HVX_VectorPred mask = Q6_Q_vcmp_gt_VsfVsf(vx, vzero_sf);

        /* Gate: vdx = mask ? vdy : 0 */
        HVX_Vector vdx = Q6_V_vmux_QVV(mask, vdy, vzero_sf);

        *(HVX_Vector *)(dx + k * 32) = vdx;
    }

    /* Scalar tail */
    int base = nvec * 32;
    for (int i = 0; i < rem; i++)
        dx[base + i] = (x[base + i] > 0.0f) ? dy[base + i] : 0.0f;
}
