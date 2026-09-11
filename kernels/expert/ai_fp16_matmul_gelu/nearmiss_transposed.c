/* Near-miss: computes A * B^T (transposed B) then applies correct GELU.
 * Numerically plausible shape but wrong matmul — must fail. */
#include "kernel_api.h"
#include <math.h>

static float gelu_f32(float x) {
    float c = 0.7978845608f * (x + 0.044715f * x * x * x);
    return 0.5f * x * (1.0f + tanhf(c));
}

void candidate_kernel(const hvx_hf *A, const hvx_hf *B, hvx_hf *out, int n) {
    for (int i = 0; i < n; i++)
        for (int j = 0; j < n; j++) {
            float acc = 0.f;
            for (int k = 0; k < n; k++) acc += (float)A[i*n+k] * (float)B[j*n+k]; /* B^T */
            hvx_hf m = (hvx_hf)acc;
            out[i*n+j] = (hvx_hf)gelu_f32((float)m);
        }
}
