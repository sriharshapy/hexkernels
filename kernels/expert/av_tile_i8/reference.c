#include <stdint.h>
/*
 * Attention A·V tile (column-major V) baseline.
 * O[i,d] = sum_j A[i*N+j] * V[d*N+j]   (int32 accumulate)
 */
void candidate_kernel(const int8_t *A, const int8_t *V, int32_t *O,
                      int M, int N, int D) {
    for (int i = 0; i < M; i++) {
        for (int d = 0; d < D; d++) {
            int32_t acc = 0;
            for (int j = 0; j < N; j++) {
                acc += (int32_t)A[i*N + j] * (int32_t)V[d*N + j];
            }
            O[i*D + d] = acc;
        }
    }
}
