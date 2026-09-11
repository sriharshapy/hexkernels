/* Near-miss: identical HVX qf16 batched RMS-norm-with-per-row-gain as
 * expert.c, but OMITS the "+1e-3" epsilon before the sqrtf (plausible bug:
 * a common, clean, in-bounds omission). Compiles, genuinely uses HVX, and
 * is correct for rows with healthy energy, but diverges (and would
 * divide-by-zero/produce inf/nan) whenever a row's mean-square is at or
 * near 0 -- the harness's row 0 is all-zero, so this MUST fail. */
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

void candidate_kernel(const hvx_hf *x, const hvx_hf *gain, hvx_hf *out, int R, int C) {
    int nb = C / BLK;
    int rem_start = nb * BLK;

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
            HVX_Vector accSq = Q6_Vqf16_vmpy_VhfVhf(v0, v0);
            for (int b = 1; b < nb; b++) {
                HVX_Vector v = xv[b];
                HVX_Vector sq = Q6_Vqf16_vmpy_VhfVhf(v, v);
                accSq = Q6_Vqf16_vadd_Vqf16Vqf16(accSq, sq);
            }
            HVX_Vector sqHf = Q6_Vhf_equals_Vqf16(accSq);
            const hvx_hf *qp = (const hvx_hf *)&sqHf;
            for (int j = 0; j < BLK; j++) sumsq += (float)qp[j];
        }
        for (int c = rem_start; c < C; c++) { float v = (float)xrow[c]; sumsq += v * v; }

        float ms = sumsq / (float)C;
        float inv_rms = 1.0f / sqrtf(ms);          /* BUG: missing "+ 1e-3f" epsilon */
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
