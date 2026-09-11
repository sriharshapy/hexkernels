/* Near-miss: forgets to saturate the accumulator before the LUT index.
   Uses (uint8_t)(acc + 128) directly, which wraps for acc outside [-128,127].
   Fails wherever acc overflows a signed byte (common with K=48 products). */
#include <stdint.h>
void candidate_kernel(const int8_t *A, const int8_t *B,
                      const int8_t *gelu_lut,
                      int8_t *out,
                      int M, int N, int K) {
    for (int i = 0; i < M; i++) {
        for (int j = 0; j < N; j++) {
            int32_t acc = 0;
            for (int k = 0; k < K; k++)
                acc += (int32_t)A[i*K+k] * (int32_t)B[k*N+j];
            /* BUG: no saturation -- wraps around for large acc values */
            uint8_t idx = (uint8_t)((int8_t)acc + 128);
            out[i*N+j] = gelu_lut[idx];
        }
    }
}
