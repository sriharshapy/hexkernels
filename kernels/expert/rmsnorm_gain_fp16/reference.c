/* PLAIN SCALAR batched fp16 RMS-norm with per-row gain -- the DENOMINATOR
 * baseline. No HVX intrinsics, no vector types anywhere. */
#include "kernel_api.h"
#include <math.h>

void candidate_kernel(const hvx_hf *x, const hvx_hf *gain, hvx_hf *out, int R, int C) {
    for (int r = 0; r < R; r++) {
        const hvx_hf *xr = x + (long)r * C;
        hvx_hf *outr = out + (long)r * C;

        float sumsq = 0.0f;
        for (int c = 0; c < C; c++) {
            float v = (float)xr[c];
            sumsq += v * v;
        }
        float ms = sumsq / (float)C;
        float inv_rms = 1.0f / sqrtf(ms + 1e-3f);
        float g = (float)gain[r];

        for (int c = 0; c < C; c++)
            outr[c] = (hvx_hf)((float)xr[c] * inv_rms * g);
    }
}
