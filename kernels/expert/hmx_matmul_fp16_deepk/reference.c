/* Scalar fp16 deep-K matmul (tolerance-correct) — the DENOMINATOR baseline.
 * Plain triple-loop float accumulate over the full K=k_dim; the HMX expert
 * must beat this by >=1.2x. */
#include "kernel_api.h"
#include "harness_common.h"
#include <stdint.h>

void candidate_kernel(const hvx_hf *A, const hvx_hf *B, hvx_hf *out, int n, int k_dim) {
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) {
            float acc = 0.f;
            for (int k = 0; k < k_dim; k++) acc += (float)A[i*k_dim+k] * (float)B[k*n+j];
            out[i*n + j] = (hvx_hf)acc;
        }
    }
}
