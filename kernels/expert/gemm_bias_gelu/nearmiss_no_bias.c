/* Near-miss: skips the bias step, applies GELU-LUT directly on saturated accumulator.
   Fails wherever bias[j] != 0 (which is almost every element). */
#include <stdint.h>
void candidate_kernel(const int8_t *A, const int8_t *B,
                      const int32_t *bias,
                      const int8_t *gelu_lut,
                      int8_t *out,
                      int M, int N, int K) {
    (void)bias;  /* BUG: bias intentionally ignored */
    for (int i = 0; i < M; i++) {
        for (int j = 0; j < N; j++) {
            int32_t acc = 0;
            for (int k = 0; k < K; k++)
                acc += (int32_t)A[i*K+k] * (int32_t)B[k*N+j];
            /* skip bias; saturate acc directly */
            if (acc >  127) acc =  127;
            if (acc < -128) acc = -128;
            int8_t pre_lut = (int8_t)acc;
            uint8_t idx = (uint8_t)(pre_lut + 128);
            out[i*N+j] = gelu_lut[idx];
        }
    }
}
