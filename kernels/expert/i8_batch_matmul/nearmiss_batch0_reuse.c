/* Near-miss: ignores batch stride — always reads A and B from batch 0.
   Fails because batches 1, 2, 3 in C will have the same (wrong) result as batch 0. */
#include <stdint.h>
#define BATCH 4
void candidate_kernel(const int8_t *A, const int8_t *B, int32_t *C,
                      int M, int N, int K) {
    for (int b = 0; b < BATCH; b++) {
        /* Bug: always uses A and B from batch 0, ignoring b*M*K / b*K*N offsets */
        int32_t *Cb = C + b * M * N;
        for (int i = 0; i < M; i++) {
            for (int j = 0; j < N; j++) {
                int32_t acc = 0;
                for (int k = 0; k < K; k++)
                    acc += (int32_t)A[i*K+k] * (int32_t)B[k*N+j]; /* batch stride missing */
                Cb[i*N+j] = acc;
            }
        }
    }
}
