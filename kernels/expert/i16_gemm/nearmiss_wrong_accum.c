/* Near-miss: accumulates in int16 instead of int32, causing overflow on large products.
   Fails because sum of 130 int16*int16 products overflows int16. */
#include <stdint.h>
void candidate_kernel(const int16_t *A, const int16_t *B, int32_t *C,
                      int M, int N, int K) {
    for (int i = 0; i < M; i++) {
        for (int j = 0; j < N; j++) {
            int16_t acc = 0;   /* wrong: int16 overflows */
            for (int k = 0; k < K; k++)
                acc += (int16_t)(A[i*K+k] * B[k*N+j]);
            C[i*N+j] = (int32_t)acc;
        }
    }
}
