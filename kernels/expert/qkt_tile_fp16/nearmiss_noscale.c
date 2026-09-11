/* Near-miss: computes the Q*K^T (column-major K) raw dot product and its
 * hf-round #1 correctly, but SKIPS the final `* scale` step entirely
 * (plausible bug: forgetting the scaling epilogue). Compiles, no HVX
 * required to be wrong -- deterministically wrong whenever scale != 1.0
 * (the harness sweeps scale=0.125, which this near-miss ignores). */
#include "kernel_api.h"

void candidate_kernel(const hvx_hf *Q, const hvx_hf *K, hvx_hf *S,
                      int M, int N, int D, _Float16 scale) {
    (void)scale;
    for (int i = 0; i < M; i++) {
        for (int j = 0; j < N; j++) {
            float acc = 0.0f;
            for (int d = 0; d < D; d++)
                acc += (float)Q[i*D + d] * (float)K[d*N + j];
            hvx_hf m = (hvx_hf)acc;
            S[i*N + j] = m;   /* BUG: missing "* scale" */
        }
    }
}
