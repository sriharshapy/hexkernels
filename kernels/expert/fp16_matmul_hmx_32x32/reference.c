/* Scalar fp16 32x32 matmul (tolerance-correct) -- the DENOMINATOR baseline.
 * Plain float32 accumulation cast to __fp16 per output element, matching the
 * harness's own scalar reference computation exactly. No HVX, no HMX. */
#include "kernel_api.h"

void candidate_kernel(const hvx_hf *A, const hvx_hf *B, hvx_hf *out, int n) {
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) {
            float acc = 0.f;
            for (int k = 0; k < n; k++) acc += (float)A[i*n+k] * (float)B[k*n+j];
            out[i*n+j] = (hvx_hf)acc;
        }
    }
}
