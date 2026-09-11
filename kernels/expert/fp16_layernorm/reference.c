/* PLAIN SCALAR fp16 layer-norm -- the DENOMINATOR baseline. No HVX
 * intrinsics, no vector types anywhere: element-at-a-time mean/var
 * reduction and per-element affine transform. */
#include "kernel_api.h"
#include <math.h>

void candidate_kernel(const hvx_hf *x, const hvx_hf *gamma, const hvx_hf *beta,
                      hvx_hf *out, int n) {
    float sumx = 0.0f, sumsq = 0.0f;
    for (int i = 0; i < n; i++) {
        float v = (float)x[i];
        sumx += v;
        sumsq += v * v;
    }
    float mean = sumx / (float)n;
    float var  = sumsq / (float)n - mean * mean;
    if (var < 0.0f) var = 0.0f;
    float inv_std = 1.0f / sqrtf(var + 1e-3f);

    for (int i = 0; i < n; i++) {
        float d = ((float)x[i] - mean) * inv_std;
        out[i] = (hvx_hf)(d * (float)gamma[i] + (float)beta[i]);
    }
}
