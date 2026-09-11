/* Scalar fp16 matmul + per-column bias (tolerance-correct) — DENOMINATOR
 * baseline. Plain triple-loop float accumulate, add bias[j], cast to __fp16
 * on store. The HMX expert must beat this by >=1.2x. */
#include "kernel_api.h"
#include "harness_common.h"
#include <stdint.h>

void candidate_kernel(const hvx_hf *A, const hvx_hf *B, const hvx_hf *bias,
                       hvx_hf *out, int n) {
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) {
            float acc = 0.f;
            for (int k = 0; k < n; k++) acc += (float)A[i*n+k] * (float)B[k*n+j];
            out[i*n + j] = (hvx_hf)(acc + (float)bias[j]);
        }
    }
}
