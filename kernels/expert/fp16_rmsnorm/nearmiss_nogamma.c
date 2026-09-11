/* Near-miss: identical HVX qf16 RMS-norm reduction as expert.c, but the
 * scale epilogue DROPS the "*gamma" step (plausible bug: normalizing but
 * forgetting the learned per-feature scale). Compiles, genuinely uses HVX,
 * correct inv_rms, but wrong whenever gamma != 1 -> must FAIL the tolerance
 * gate (gamma is pinned != 1 for most indices). */
#include "kernel_api.h"
#include "harness_common.h"
#include <hexagon_types.h>
#include <hexagon_protos.h>
#include <math.h>

#define BLK 64

static inline HVX_Vector hf_splat_f(float f) {
    hvx_hf v = (hvx_hf)f;
    unsigned short bits = *(const unsigned short *)&v;
    return Q6_Vh_vsplat_R((int)bits);
}

void candidate_kernel(const hvx_hf *x, const hvx_hf *gamma, hvx_hf *out, int n) {
    int nb = n / BLK;

    const HVX_Vector *xv = (const HVX_Vector *)x;
    HVX_Vector v0 = xv[0];
    HVX_Vector accSq = Q6_Vqf16_vmpy_VhfVhf(v0, v0);
    for (int b = 1; b < nb; b++) {
        HVX_Vector v = xv[b];
        HVX_Vector sq = Q6_Vqf16_vmpy_VhfVhf(v, v);
        accSq = Q6_Vqf16_vadd_Vqf16Vqf16(accSq, sq);
    }
    HVX_Vector sqHf = Q6_Vhf_equals_Vqf16(accSq);
    const hvx_hf *qp = (const hvx_hf *)&sqHf;

    float sumsq = 0.0f;
    for (int j = 0; j < BLK; j++) sumsq += (float)qp[j];

    float ms = sumsq / (float)n;
    float inv_rms = 1.0f / sqrtf(ms + 1e-3f);

    HVX_Vector invrmsVec = hf_splat_f(inv_rms);

    HVX_Vector *ov = (HVX_Vector *)out;
    for (int b = 0; b < nb; b++) {
        HVX_Vector v = xv[b];
        HVX_Vector sc = Q6_Vqf16_vmpy_VhfVhf(v, invrmsVec);
        ov[b] = Q6_Vhf_equals_Vqf16(sc);   /* bug: never multiplies by gamma */
    }
    (void)gamma;
}
