/* Near-miss: ignores the runtime gelu_lut; outputs the saturated accumulator
   directly (identity activation) instead of doing the LUT lookup.
   Fails on virtually every element. */
#include <stdint.h>
void candidate_kernel(const int8_t *A, const int8_t *B,
                      const int8_t *gelu_lut,
                      int8_t *out,
                      int M, int N, int K) {
    (void)gelu_lut;  /* BUG: LUT ignored */
    for (int i = 0; i < M; i++) {
        for (int j = 0; j < N; j++) {
            int32_t acc = 0;
            for (int k = 0; k < K; k++)
                acc += (int32_t)A[i*K+k] * (int32_t)B[k*N+j];
            int32_t sat = acc;
            if (sat >  127) sat =  127;
            if (sat < -128) sat = -128;
            /* BUG: returns saturated acc instead of gelu_lut[idx] */
            out[i*N+j] = (int8_t)sat;
        }
    }
}
