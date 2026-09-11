#include <stdint.h>
void candidate_kernel(const int8_t *A, const int8_t *B,
                      const int8_t *gelu_lut,
                      int8_t *out,
                      int M, int N, int K) {
    for (int i = 0; i < M; i++) {
        for (int j = 0; j < N; j++) {
            /* Step 1: matmul accumulate */
            int32_t acc = 0;
            for (int k = 0; k < K; k++)
                acc += (int32_t)A[i*K+k] * (int32_t)B[k*N+j];
            /* Step 2: saturate to int8 */
            int32_t sat = acc;
            if (sat >  127) sat =  127;
            if (sat < -128) sat = -128;
            int8_t pre_lut = (int8_t)sat;
            /* Step 3: GELU via LUT */
            uint8_t idx = (uint8_t)(pre_lut + 128);
            out[i*N+j] = gelu_lut[idx];
        }
    }
}
