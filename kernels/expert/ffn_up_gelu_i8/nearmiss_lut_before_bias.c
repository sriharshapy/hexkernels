/* Near-miss: applies the GELU LUT lookup BEFORE adding bias (order-swap).
 *   pre = clamp(acc,-128,127); g = gelu_lut[pre+128]; out = clamp(g+bias[j])
 * instead of the correct
 *   biased = clamp(acc+bias[j],-128,127); out = gelu_lut[biased+128]
 * A very plausible, deterministic bug -- wrong whenever bias[j] != 0. */
#include <stdint.h>
void candidate_kernel(const int8_t *A, const int8_t *B,
                      const int32_t *bias, const int8_t *gelu_lut,
                      int8_t *out, int M, int N, int K) {
    for (int i = 0; i < M; i++) {
        for (int j = 0; j < N; j++) {
            int32_t acc = 0;
            for (int k = 0; k < K; k++)
                acc += (int32_t)A[i*K+k] * (int32_t)B[j*K+k];
            /* BUG: saturate + LUT BEFORE bias, not after */
            int32_t sat = acc;
            if (sat >  127) sat =  127;
            if (sat < -128) sat = -128;
            int8_t pre = (int8_t)sat;
            uint8_t idx = (uint8_t)(pre + 128);
            int32_t g = gelu_lut[idx];
            int64_t withbias = (int64_t)g + (int64_t)bias[j];
            if (withbias >  127) withbias =  127;
            if (withbias < -128) withbias = -128;
            out[i*N+j] = (int8_t)withbias;
        }
    }
}
