/* Scalar fp16 attention A.V with deep Skv=128 reduction -- the DENOMINATOR
 * baseline. Plain float32 accumulation cast to __fp16 per output element,
 * matching the harness's own scalar reference computation exactly. No HVX,
 * no HMX. */
#include "kernel_api.h"

void candidate_kernel(const hvx_hf *P, const hvx_hf *V, hvx_hf *out, int n, int k_dim) {
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) {
            float acc = 0.f;
            for (int k = 0; k < k_dim; k++) acc += (float)P[i*k_dim+k] * (float)V[k*n+j];
            out[i*n+j] = (hvx_hf)acc;
        }
    }
}
