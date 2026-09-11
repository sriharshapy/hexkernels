/* sol_03: HVX-vectorized MSE loss using Q6_Vqf32_* qfloat intrinsics.
 * Processes 32 fp32 elements/cycle via HVX 128-byte vectors (4 bytes each = 32 lanes).
 * Semantics: out[0] = (1/n) * sum_i (pred[i] - tgt[i])^2 */
#include "kernel_api.h"
#include <hexagon_types.h>
#include <hexagon_protos.h>

void candidate_kernel(const float *pred, const float *tgt, float *out, int n) {
    /* Accumulate partial sums in a qf32 vector, then horizontal-reduce. */
    HVX_Vector vacc = Q6_Vqf32_vadd_Vqf32Vqf32(
        Q6_V_vzero(), Q6_V_vzero());  /* zero qf32 accumulator */

    int nvec = n / 32;           /* number of full 128-byte vectors */
    int rem  = n - nvec * 32;

    for (int k = 0; k < nvec; k++) {
        HVX_Vector vp = *(const HVX_Vector *)(pred + k * 32);
        HVX_Vector vt = *(const HVX_Vector *)(tgt  + k * 32);
        /* diff = pred - tgt (convert to qf32 first) */
        HVX_Vector vd = Q6_Vqf32_vsub_VsfVsf(vp, vt);
        /* diff^2 */
        HVX_Vector vd2 = Q6_Vqf32_vmpy_Vqf32Vqf32(vd, vd);
        vacc = Q6_Vqf32_vadd_Vqf32Vqf32(vacc, vd2);
    }

    /* Horizontal reduce: fold 32 lanes down to scalar. */
    /* Convert accumulated qf32 vector back to sf for partial-sum extraction. */
    HVX_Vector vsf = Q6_Vsf_equals_Vqf32(vacc);

    /* Fold using HVX vdelta/deal into scalar partial sums. */
    /* We do a tree-reduction by extracting pairs and summing them. */
    float tmp[32] __attribute__((aligned(128)));
    *(HVX_Vector *)tmp = vsf;
    float hsum = 0.0f;
    for (int i = 0; i < 32; i++) hsum += tmp[i];

    /* Scalar tail */
    float tail = 0.0f;
    int base = nvec * 32;
    for (int i = 0; i < rem; i++) {
        float d = pred[base + i] - tgt[base + i];
        tail += d * d;
    }

    out[0] = (hsum + tail) / (float)n;
}
