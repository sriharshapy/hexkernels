/* Near-miss: indexes V as if it were [N x D] row-major (V[j*D+d]) instead
 * of the actual [D x N] column-major layout (V[d*N+j]). Same idea as the
 * sibling attention_av_tile task's nearmiss_transposed_V.c. Compiles and
 * stays in-bounds (D*N = N*D), but reads the wrong bytes whenever D != N
 * (here D=90, N=16), producing incorrect output. */
#include <stdint.h>
void candidate_kernel(const int8_t *A, const int8_t *V, int32_t *O,
                      int M, int N, int D) {
    for (int i = 0; i < M; i++) {
        for (int d = 0; d < D; d++) {
            int32_t acc = 0;
            for (int j = 0; j < N; j++)
                acc += (int32_t)A[i*N+j] * (int32_t)V[j*D+d];  /* wrong index */
            O[i*D+d] = acc;
        }
    }
}
