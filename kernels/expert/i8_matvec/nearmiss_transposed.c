/* Near-miss: uses transposed indexing A[k*M+i] instead of A[i*K+k].
   Fails because rows and columns are swapped, producing wrong dot products. */
#include <stdint.h>
void candidate_kernel(const int8_t *A, const int8_t *x, int32_t *y,
                      int M, int K) {
    for (int i = 0; i < M; i++) {
        int32_t acc = 0;
        for (int k = 0; k < K; k++)
            acc += (int32_t)A[k*M+i] * (int32_t)x[k];  /* transposed: k*M+i instead of i*K+k */
        y[i] = acc;
    }
}
