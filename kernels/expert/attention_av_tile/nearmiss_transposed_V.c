/* Near-miss: indexes V as transposed — V[d*N+j] instead of V[j*D+d].
 * This reads the wrong memory layout, producing incorrect output when D != N
 * (here D=130, N=16 so this is clearly wrong). */
#include <stdint.h>
void candidate_kernel(const int8_t *A, const int8_t *V, int32_t *O,
                      int M, int N, int D) {
    /* BUG: V indexed as [D x N] transposed instead of [N x D] */
    for (int i = 0; i < M; i++) {
        for (int d = 0; d < D; d++) {
            int32_t acc = 0;
            for (int j = 0; j < N; j++)
                acc += (int32_t)A[i*N+j] * (int32_t)V[d*N+j];  /* wrong index */
            O[i*D+d] = acc;
        }
    }
}
