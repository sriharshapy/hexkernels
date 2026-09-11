/* Near-miss: truncates instead of rounding half-away-from-zero.
   Fails on rounding-sensitive cases (e.g. when remainder is exactly half). */
#include <stdint.h>
void candidate_kernel(const int8_t *A, const int8_t *B, int8_t *out,
                      int M, int N, int K,
                      int32_t mult, int shift, int8_t zp) {
    for (int i = 0; i < M; i++) {
        for (int j = 0; j < N; j++) {
            int32_t acc = 0;
            for (int k = 0; k < K; k++)
                acc += (int32_t)A[i*K+k] * (int32_t)B[k*N+j];
            /* truncation instead of round-half-away-from-zero */
            long long v = (long long)acc * (long long)mult;
            long long r = v >> shift;   /* truncates — wrong for ties */
            r += zp;
            if (r > 127) r = 127; if (r < -128) r = -128;
            out[i*N+j] = (signed char)r;
        }
    }
}
