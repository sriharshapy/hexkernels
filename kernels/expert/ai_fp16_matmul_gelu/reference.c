/* Genuine scalar baseline (the denominator): plain triple-nested-loop
 * matmul in float, fp16-round to match the HMX expert's natural output
 * precision (so both are being held to the identical harness reference),
 * then GELU (tanh approximation) in float, final fp16 cast. No HVX, no
 * HMX -- single-engine scalar. (The v5-inherited file at this path used to
 * BE the HMX kernel, mis-tagged mechanisms:["scalar"] -- that kernel is now
 * expert.c, and this genuinely-scalar file is the new denominator.) */
#include "kernel_api.h"
#include <math.h>

static float gelu_f32(float x) {
    float c = 0.7978845608f * (x + 0.044715f * x * x * x);
    return 0.5f * x * (1.0f + tanhf(c));
}

void candidate_kernel(const hvx_hf *A, const hvx_hf *B, hvx_hf *out, int n) {
    for (int i = 0; i < n; i++)
        for (int j = 0; j < n; j++) {
            float acc = 0.0f;
            for (int k = 0; k < n; k++)
                acc += (float)A[i*n+k] * (float)B[k*n+j];
            hvx_hf m = (hvx_hf)acc;   /* fp16-round to match HMX's natural output */
            out[i*n+j] = (hvx_hf)gelu_f32((float)m);
        }
}
