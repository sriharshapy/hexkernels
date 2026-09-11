/* PLAIN SCALAR fp16 32x32x32 matmul + bias + GELU -- the DENOMINATOR
 * baseline. No HVX intrinsics, no vector types anywhere (matmul AND GELU
 * epilogue both scalar). */
#include "kernel_api.h"
#include <math.h>

static float gelu_f32(float x) {
    float c = 0.7978845608f * (x + 0.044715f * x * x * x);
    return 0.5f * x * (1.0f + tanhf(c));
}

void candidate_kernel(const hvx_hf *A, const hvx_hf *B, const hvx_hf *bias,
                      hvx_hf *out, int n, int k_dim) {
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) {
            float acc = 0.0f;
            for (int k = 0; k < k_dim; k++)
                acc += (float)A[i*k_dim + k] * (float)B[k*n + j];
            hvx_hf m = (hvx_hf)acc;
            float t = (float)m + (float)bias[j];
            out[i*n + j] = (hvx_hf)gelu_f32(t);
        }
    }
}
