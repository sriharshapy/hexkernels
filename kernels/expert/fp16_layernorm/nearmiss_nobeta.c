/* Near-miss: identical HVX qf16 layer-norm reduction as expert.c, but the
 * affine epilogue DROPS the "+beta" step (plausible bug: confusing a
 * gamma-only scale with the full affine transform). Compiles, genuinely
 * uses HVX, correct mean/var/gamma-scale, but wrong whenever beta != 0 ->
 * must FAIL the tolerance gate (beta is pinned nonzero for most indices). */
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

void candidate_kernel(const hvx_hf *x, const hvx_hf *gamma, const hvx_hf *beta,
                      hvx_hf *out, int n) {
    int nb = n / BLK;

    const HVX_Vector *xv = (const HVX_Vector *)x;
    HVX_Vector v0 = xv[0];
    HVX_Vector accSum = Q6_Vqf16_vadd_VhfVhf(v0, hf_splat_f(0.0f));
    HVX_Vector accSq  = Q6_Vqf16_vmpy_VhfVhf(v0, v0);
    for (int b = 1; b < nb; b++) {
        HVX_Vector v = xv[b];
        HVX_Vector sq = Q6_Vqf16_vmpy_VhfVhf(v, v);
        accSum = Q6_Vqf16_vadd_Vqf16Vhf(accSum, v);
        accSq  = Q6_Vqf16_vadd_Vqf16Vqf16(accSq, sq);
    }
    HVX_Vector sumHf = Q6_Vhf_equals_Vqf16(accSum);
    HVX_Vector sqHf  = Q6_Vhf_equals_Vqf16(accSq);
    const hvx_hf *sp = (const hvx_hf *)&sumHf;
    const hvx_hf *qp = (const hvx_hf *)&sqHf;

    float sumx = 0.0f, sumsq = 0.0f;
    for (int j = 0; j < BLK; j++) { sumx += (float)sp[j]; sumsq += (float)qp[j]; }

    float mean = sumx / (float)n;
    float var  = sumsq / (float)n - mean * mean;
    if (var < 0.0f) var = 0.0f;
    float inv_std = 1.0f / sqrtf(var + 1e-3f);

    HVX_Vector meanVec   = hf_splat_f(mean);
    HVX_Vector invstdVec = hf_splat_f(inv_std);

    const HVX_Vector *gv = (const HVX_Vector *)gamma;
    HVX_Vector *ov = (HVX_Vector *)out;
    for (int b = 0; b < nb; b++) {
        HVX_Vector v = xv[b];
        HVX_Vector d  = Q6_Vqf16_vsub_VhfVhf(v, meanVec);
        HVX_Vector dh = Q6_Vhf_equals_Vqf16(d);
        HVX_Vector sc  = Q6_Vqf16_vmpy_VhfVhf(dh, invstdVec);
        HVX_Vector sch = Q6_Vhf_equals_Vqf16(sc);
        HVX_Vector gp  = Q6_Vqf16_vmpy_VhfVhf(sch, gv[b]);
        ov[b] = Q6_Vhf_equals_Vqf16(gp);   /* bug: never adds beta */
    }
}
