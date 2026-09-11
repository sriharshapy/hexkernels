/* EXPERT (achievability bar) = solutions/s2.c (vector-native ror-shift
 * horizontal reduce for sum-of-squares), chosen over solutions/s1.c
 * (extract-and-sum scalar loop over the block's 64 lanes) by measurement:
 * s2 steady-state 9798 kernel cycles vs s1's 16047 -- the scalar loop over
 * all 64 lanes per row costs more than the 6-step ror-shift butterfly. */
#include "kernel_api.h"
#include "harness_common.h"
#include <hexagon_types.h>
#include <hexagon_protos.h>
#include <math.h>

#define BLK 64
#define ROWCAP 128

static inline HVX_Vector hf_splat_f(float f) {
    hvx_hf v = (hvx_hf)f;
    unsigned short bits = *(const unsigned short *)&v;
    return Q6_Vh_vsplat_R((int)bits);
}

static inline HVX_Vector hreduce_qf16(HVX_Vector v) {
    v = Q6_Vqf16_vadd_Vqf16Vqf16(v, Q6_V_vror_VR(v, 64));
    v = Q6_Vqf16_vadd_Vqf16Vqf16(v, Q6_V_vror_VR(v, 32));
    v = Q6_Vqf16_vadd_Vqf16Vqf16(v, Q6_V_vror_VR(v, 16));
    v = Q6_Vqf16_vadd_Vqf16Vqf16(v, Q6_V_vror_VR(v, 8));
    v = Q6_Vqf16_vadd_Vqf16Vqf16(v, Q6_V_vror_VR(v, 4));
    v = Q6_Vqf16_vadd_Vqf16Vqf16(v, Q6_V_vror_VR(v, 2));
    return v;
}

void candidate_kernel(const hvx_hf *x, const hvx_hf *gain, hvx_hf *out, int R, int C) {
    int nb = C / BLK;              /* 80/64 = 1 full block */
    int rem_start = nb * BLK;      /* 64 */

    for (int r = 0; r < R; r++) {
        const hvx_hf *xr = x + (long)r * C;
        hvx_hf *outr = out + (long)r * C;

        hvx_hf xrow[ROWCAP] __attribute__((aligned(128)));
        hvx_hf orow[ROWCAP] __attribute__((aligned(128)));
        for (int c = 0; c < C; c++) xrow[c] = xr[c];

        const HVX_Vector *xv = (const HVX_Vector *)xrow;
        float sumsq = 0.0f;

        if (nb > 0) {
            HVX_Vector v0 = xv[0];
            HVX_Vector sq = Q6_Vqf16_vmpy_VhfVhf(v0, v0);
            HVX_Vector red = hreduce_qf16(sq);
            HVX_Vector redHf = Q6_Vhf_equals_Vqf16(red);
            const hvx_hf *rp = (const hvx_hf *)&redHf;
            sumsq += (float)rp[0];
        }
        for (int c = rem_start; c < C; c++) { float v = (float)xrow[c]; sumsq += v * v; }

        float ms = sumsq / (float)C;
        float inv_rms = 1.0f / sqrtf(ms + 1e-3f);
        float combined = inv_rms * (float)gain[r];
        HVX_Vector combinedVec = hf_splat_f(combined);

        HVX_Vector *ov = (HVX_Vector *)orow;
        for (int b = 0; b < nb; b++) {
            HVX_Vector v = xv[b];
            HVX_Vector sc = Q6_Vqf16_vmpy_VhfVhf(v, combinedVec);
            ov[b] = Q6_Vhf_equals_Vqf16(sc);
        }
        for (int c = rem_start; c < C; c++)
            orow[c] = (hvx_hf)((float)xrow[c] * combined);

        for (int c = 0; c < C; c++) outr[c] = orow[c];
    }
}
