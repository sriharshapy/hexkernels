/* PLAIN SCALAR fp16 QK^T tile (column-major K) -- the DENOMINATOR baseline.
 * No HVX intrinsics, no vector types anywhere. */
#include "kernel_api.h"

void candidate_kernel(const hvx_hf *Q, const hvx_hf *K, hvx_hf *S,
                      int M, int N, int D, _Float16 scale) {
    for (int i = 0; i < M; i++) {
        for (int j = 0; j < N; j++) {
            float acc = 0.0f;
            for (int d = 0; d < D; d++)
                acc += (float)Q[i*D + d] * (float)K[d*N + j];
            hvx_hf m = (hvx_hf)acc;
            S[i*N + j] = (hvx_hf)((float)m * (float)scale);
        }
    }
}
