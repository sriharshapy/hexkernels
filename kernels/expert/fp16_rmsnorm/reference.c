/* PLAIN SCALAR fp16 RMS-norm -- the DENOMINATOR baseline. No HVX
 * intrinsics, no vector types anywhere. */
#include "kernel_api.h"
#include <math.h>

void candidate_kernel(const hvx_hf *x, const hvx_hf *gamma, hvx_hf *out, int n) {
    float sumsq = 0.0f;
    for (int i = 0; i < n; i++) {
        float v = (float)x[i];
        sumsq += v * v;
    }
    float ms = sumsq / (float)n;
    float inv_rms = 1.0f / sqrtf(ms + 1e-3f);

    for (int i = 0; i < n; i++)
        out[i] = (hvx_hf)((float)x[i] * inv_rms * (float)gamma[i]);
}
