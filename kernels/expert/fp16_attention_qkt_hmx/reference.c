/* Scalar fp16 attention QK^T with deep D=128 reduction -- the DENOMINATOR
 * baseline. Plain float32 accumulation cast to __fp16 per output element,
 * matching the harness's own scalar reference computation exactly. No HVX,
 * no HMX. K is stored [S,D] row-major (row j IS key vector j). */
#include "kernel_api.h"

void candidate_kernel(const hvx_hf *Q, const hvx_hf *K, hvx_hf *out, int n, int k_dim) {
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) {
            float acc = 0.f;
            for (int d = 0; d < k_dim; d++) acc += (float)Q[i*k_dim+d] * (float)K[j*k_dim+d];
            out[i*n+j] = (hvx_hf)acc;
        }
    }
}
