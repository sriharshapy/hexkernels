/* HVX qf16 layer-norm -- the ACCELERATED expert (pure HVX, no matrix engine).
 * Pass 1 (reduction): accumulate sum(x) and sum(x^2) as 64-lane qf16
 * accumulator vectors across the n/64 blocks (Q6_Vqf16_vadd_Vqf16Vqf16 /
 * Q6_Vqf16_vmpy_VhfVhf), then unpack BOTH 64-lane accumulators to scalar
 * once and finish the horizontal sum in float (64 adds -- negligible next
 * to the n-element vectorized reduction). mean/var/inv_std are then plain
 * scalars (a single scalar sqrtf call, O(1), not O(n)).
 * Pass 2 (affine): (x-mean)*inv_std*gamma+beta fused as native HVX qf16
 * vector ops per block -- mean/inv_std are splatted (bit-pattern broadcast,
 * the same idiom used by the matmul row-broadcast siblings), gamma/beta are
 * loaded per-block since they vary per feature index. */
#include "kernel_api.h"
#include "harness_common.h"
#include <hexagon_types.h>
#include <hexagon_protos.h>
#include <math.h>

#define BLK 64   /* fp16 lanes per 128B HVX vector */

/* NOTE: __fp16 cannot be a function parameter on this target ("parameters
 * cannot have __fp16 type") -- take a plain float and convert to hvx_hf in a
 * LOCAL variable (locals are fine, only by-value parameters are rejected). */
static inline HVX_Vector hf_splat_f(float f) {
    hvx_hf v = (hvx_hf)f;
    unsigned short bits = *(const unsigned short *)&v;
    return Q6_Vh_vsplat_R((int)bits);
}

void candidate_kernel(const hvx_hf *x, const hvx_hf *gamma, const hvx_hf *beta,
                      hvx_hf *out, int n) {
    int nb = n / BLK;   /* n is a multiple of BLK (single-tile, no remainder) */

    const HVX_Vector *xv = (const HVX_Vector *)x;
    HVX_Vector v0 = xv[0];
    HVX_Vector accSum = Q6_Vqf16_vadd_VhfVhf(v0, hf_splat_f(0.0f));  /* hf -> qf16 */
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
    if (var < 0.0f) var = 0.0f;   /* guard tiny negative from fp roundoff */
    float inv_std = 1.0f / sqrtf(var + 1e-3f);

    HVX_Vector meanVec   = hf_splat_f(mean);
    HVX_Vector invstdVec = hf_splat_f(inv_std);

    const HVX_Vector *gv = (const HVX_Vector *)gamma;
    const HVX_Vector *bv = (const HVX_Vector *)beta;
    HVX_Vector *ov = (HVX_Vector *)out;
    for (int b = 0; b < nb; b++) {
        HVX_Vector v = xv[b];
        HVX_Vector d  = Q6_Vqf16_vsub_VhfVhf(v, meanVec);          /* x - mean */
        HVX_Vector dh = Q6_Vhf_equals_Vqf16(d);
        HVX_Vector sc  = Q6_Vqf16_vmpy_VhfVhf(dh, invstdVec);       /* * inv_std */
        HVX_Vector sch = Q6_Vhf_equals_Vqf16(sc);
        HVX_Vector gp  = Q6_Vqf16_vmpy_VhfVhf(sch, gv[b]);          /* * gamma */
        HVX_Vector gph = Q6_Vhf_equals_Vqf16(gp);
        HVX_Vector oq  = Q6_Vqf16_vadd_VhfVhf(gph, bv[b]);          /* + beta */
        ov[b] = Q6_Vhf_equals_Vqf16(oq);
    }
}
