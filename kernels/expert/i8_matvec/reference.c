#include <stdint.h>
void candidate_kernel(const int8_t *A, const int8_t *x, int32_t *y,
                      int M, int K) {
    for (int i = 0; i < M; i++) {
        int32_t acc = 0;
        for (int k = 0; k < K; k++)
            acc += (int32_t)A[i*K+k] * (int32_t)x[k];
        y[i] = acc;
    }
}
