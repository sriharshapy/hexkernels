/* Near-miss: uses the wrong bias stride -- indexes bias[j] (batch 0's bias for all
   batches) instead of bias[b*N+j]. All batches get the same bias, so batches 1-3 fail. */
#include <stdint.h>
#define BATCH 4
void candidate_kernel(const int8_t *A, const int8_t *B,
                      const int32_t *bias, int32_t *C,
                      int M, int N, int K) {
    for (int b = 0; b < BATCH; b++) {
        const int8_t *Ab = A + b * M * K;
        const int8_t *Bb = B + b * K * N;
        int32_t      *Cb = C + b * M * N;
        for (int i = 0; i < M; i++) {
            for (int j = 0; j < N; j++) {
                int32_t acc = 0;
                for (int k = 0; k < K; k++)
                    acc += (int32_t)Ab[i*K+k] * (int32_t)Bb[k*N+j];
                /* BUG: bias[j] instead of bias[b*N+j] -- always uses batch-0 bias */
                Cb[i*N+j] = acc + bias[j];
            }
        }
    }
}
