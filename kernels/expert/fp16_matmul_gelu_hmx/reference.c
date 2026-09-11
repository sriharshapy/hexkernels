/* PLAIN SCALAR fp16 32x32 matmul + GELU -- the DENOMINATOR baseline. No HVX
 * intrinsics, no vector types anywhere (matmul AND GELU epilogue both scalar). */
#include "kernel_api.h"
#include <math.h>

static float gelu_f32(float x) {
    float c = 0.7978845608f * (x + 0.044715f * x * x * x);
    return 0.5f * x * (1.0f + tanhf(c));
}

void candidate_kernel(const hvx_hf *A, const hvx_hf *B, hvx_hf *out, int n) {
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) {
            float acc = 0.0f;
            for (int k = 0; k < n; k++)
                acc += (float)A[i*n + k] * (float)B[k*n + j];
            hvx_hf m = (hvx_hf)acc;
            out[i*n + j] = (hvx_hf)gelu_f32((float)m);
        }
    }
}
