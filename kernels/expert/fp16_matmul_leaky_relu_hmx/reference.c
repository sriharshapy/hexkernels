/* Scalar fp16 32x32x128 (deep-K) matmul + leaky ReLU (tolerance-correct) --
 * the DENOMINATOR baseline. Plain float32 accumulation cast to __fp16
 * (matching the HMX-rounded intermediate), then leaky ReLU (slope 1/8) in
 * float, cast back to __fp16 -- matching the harness's own scalar reference
 * computation exactly. No HVX, no HMX. */
#include "kernel_api.h"

void candidate_kernel(const hvx_hf *A, const hvx_hf *B, hvx_hf *out, int n, int k_dim) {
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) {
            float acc = 0.f;
            for (int k = 0; k < k_dim; k++) acc += (float)A[i*k_dim+k] * (float)B[k*n+j];
            hvx_hf m = (hvx_hf)acc;
            float v = (float)m;
            out[i*n+j] = (hvx_hf)(v > 0.0f ? v : v * 0.125f);
        }
    }
}
