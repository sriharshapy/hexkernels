/* Near-miss: ignores the runtime gelu_lut and instead applies a fixed identity LUT.
   Passes the LUT indices correctly but uses the wrong table.
   Fails on every element where gelu_lut != identity (virtually everywhere). */
#include <stdint.h>
void candidate_kernel(const int8_t *A, const int8_t *B,
                      const int32_t *bias,
                      const int8_t *gelu_lut,
                      int8_t *out,
                      int M, int N, int K) {
    (void)gelu_lut;  /* BUG: ignores runtime LUT, uses identity mapping instead */
    for (int i = 0; i < M; i++) {
        for (int j = 0; j < N; j++) {
            int32_t acc = 0;
            for (int k = 0; k < K; k++)
                acc += (int32_t)A[i*K+k] * (int32_t)B[k*N+j];
            int64_t biased = (int64_t)acc + (int64_t)bias[j];
            if (biased >  127) biased =  127;
            if (biased < -128) biased = -128;
            /* BUG: output the saturated value directly instead of LUT lookup */
            out[i*N+j] = (int8_t)biased;
        }
    }
}
