#include <stdint.h>
void candidate_kernel(const int8_t *A, const int8_t *B, const int32_t *bias,
                      int8_t *out, int M, int N, int K,
                      int32_t scale_mult, int scale_shift) {
    for (int i = 0; i < M; i++) {
        for (int j = 0; j < N; j++) {
            int32_t acc = 0;
            for (int k = 0; k < K; k++)
                acc += (int32_t)A[i*K+k] * (int32_t)B[j*K+k];
            int64_t biased = (int64_t)acc + (int64_t)bias[j];
            int64_t r    = biased * (int64_t)scale_mult;
            int64_t half = (scale_shift > 0) ? ((int64_t)1 << (scale_shift - 1)) : 0;
            int64_t q    = (r >= 0) ? ((r + half) >> scale_shift)
                                    : -(((-r) + half) >> scale_shift);
            if (q >  127) q =  127;
            if (q < -128) q = -128;
            out[i*N+j] = (int8_t)q;
        }
    }
}
