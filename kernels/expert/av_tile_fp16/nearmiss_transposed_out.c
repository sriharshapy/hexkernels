/* Near-miss: computes the correct dot product for every (i,d) pair, but
 * writes it to the TRANSPOSED output slot O[d*M+i] instead of O[i*D+d]
 * (plausible index-swap bug). Since M=D=8 here, both indexings stay
 * in-bounds (no OOB), but the write pattern is wrong whenever the M x D
 * result matrix is not symmetric under transpose -- true with overwhelming
 * probability for random A, V data. */
#include "kernel_api.h"

void candidate_kernel(const hvx_hf *A, const hvx_hf *V, hvx_hf *O,
                      int M, int N, int D) {
    for (int i = 0; i < M; i++) {
        for (int d = 0; d < D; d++) {
            float acc = 0.0f;
            for (int j = 0; j < N; j++)
                acc += (float)A[i*N + j] * (float)V[d*N + j];
            O[d*M + i] = (hvx_hf)acc;   /* BUG: should be O[i*D + d] */
        }
    }
}
