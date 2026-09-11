/* Near-miss: ignores the bias vector, just stores the raw matmul result.
   Fails because bias[j] is never added to each column. */
#include <stdint.h>
void candidate_kernel(const int8_t *A, const int8_t *B,
                      const int32_t *bias, int32_t *C,
                      int M, int N, int K) {
    for (int i = 0; i < M; i++) {
        for (int j = 0; j < N; j++) {
            int32_t acc = 0;
            for (int k = 0; k < K; k++)
                acc += (int32_t)A[i*K+k] * (int32_t)B[k*N+j];
            C[i*N+j] = acc;   /* bias[j] omitted — wrong */
        }
    }
}
