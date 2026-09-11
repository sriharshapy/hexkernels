/* PLAIN SCALAR fp16 32x32x128 (deep-K) matmul + per-column bias add + ReLU --
 * the DENOMINATOR baseline. No HVX intrinsics, no vector types: a competent
 * single-element-at-a-time C loop, float-accumulate then fp16-round to match
 * the expert's numeric path. */
#include "kernel_api.h"

void candidate_kernel(const hvx_hf *A, const hvx_hf *B, const hvx_hf *bias,
                      hvx_hf *out, int n, int k_dim) {
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) {
            float acc = 0.0f;
            for (int k = 0; k < k_dim; k++)
                acc += (float)A[i*k_dim + k] * (float)B[k*n + j];
            hvx_hf m = (hvx_hf)acc;                       /* fp16-round */
            float t = (float)m + (float)bias[j];
            out[i*n + j] = (hvx_hf)(t < 0.0f ? 0.0f : t);
        }
    }
}
