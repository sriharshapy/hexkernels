/* PLAIN SCALAR fp16 A.V tile (column-major V) -- the DENOMINATOR baseline.
 * No HVX intrinsics, no vector types anywhere. */
#include "kernel_api.h"

void candidate_kernel(const hvx_hf *A, const hvx_hf *V, hvx_hf *O,
                      int M, int N, int D) {
    for (int i = 0; i < M; i++) {
        for (int d = 0; d < D; d++) {
            float acc = 0.0f;
            for (int j = 0; j < N; j++)
                acc += (float)A[i*N + j] * (float)V[d*N + j];
            O[i*D + d] = (hvx_hf)acc;
        }
    }
}
