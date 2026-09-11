/* HVX qf16 RMS-norm -- the ACCELERATED expert (pure HVX, no matrix engine).
 * Pass 1 (reduction): accumulate sum(x^2) as a 64-lane qf16 accumulator
 * vector across the n/64 blocks, unpack ONCE and finish the horizontal sum
 * in scalar float (64 adds -- negligible next to the vectorized n-element
 * reduction). inv_rms is then a plain scalar (one scalar sqrtf call, O(1)).
 * Pass 2 (scale): x*inv_rms*gamma fused as native HVX qf16 vector ops per
 * block -- inv_rms is splatted (bit-pattern broadcast), gamma is loaded
 * per-block since it varies per feature index. */
#include "kernel_api.h"
#include "harness_common.h"
#include <hexagon_types.h>
#include <hexagon_protos.h>
#include <math.h>

#define BLK 64   /* fp16 lanes per 128B HVX vector */

/* NOTE: __fp16 cannot be a function parameter on this target -- take a
 * plain float and convert to hvx_hf in a LOCAL variable. */
static inline HVX_Vector hf_splat_f(float f) {
    hvx_hf v = (hvx_hf)f;
    unsigned short bits = *(const unsigned short *)&v;
    return Q6_Vh_vsplat_R((int)bits);
}

void candidate_kernel(const hvx_hf *x, const hvx_hf *gamma, hvx_hf *out, int n) {
    int nb = n / BLK;   /* n is a multiple of BLK (single-tile, no remainder) */

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

    const HVX_Vector *gv = (const HVX_Vector *)gamma;
    HVX_Vector *ov = (HVX_Vector *)out;
    for (int b = 0; b < nb; b++) {
        HVX_Vector v = xv[b];
        HVX_Vector sc  = Q6_Vqf16_vmpy_VhfVhf(v, invrmsVec);   /* x * inv_rms */
        HVX_Vector sch = Q6_Vhf_equals_Vqf16(sc);
        HVX_Vector gp  = Q6_Vqf16_vmpy_VhfVhf(sch, gv[b]);      /* * gamma */
        ov[b] = Q6_Vhf_equals_Vqf16(gp);
    }
}
