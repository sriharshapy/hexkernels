#include <stdint.h>
void candidate_kernel(const int8_t *A, const int8_t *B,
                      const int32_t *bias, const int8_t *gelu_lut,
                      int8_t *out, int M, int N, int K) {
    for (int i = 0; i < M; i++) {
        for (int j = 0; j < N; j++) {
            int32_t acc = 0;
            for (int k = 0; k < K; k++)
                acc += (int32_t)A[i*K+k] * (int32_t)B[j*K+k];
            int64_t biased = (int64_t)acc + (int64_t)bias[j];
            if (biased >  127) biased =  127;
            if (biased < -128) biased = -128;
            int8_t pre = (int8_t)biased;
            uint8_t idx = (uint8_t)(pre + 128);
            out[i*N+j] = gelu_lut[idx];
        }
    }
}
