/* Near-miss: computes Q*K^T (column-major K) correctly, but requantizes
 * with a plain TRUNCATING right shift instead of round-half-away-from-zero.
 * Compiles and is "close" but drifts by up to 0.5 LSB after the >>scale_shift
 * -- wrong whenever scale_shift > 0 and the discarded bits are >= half. */
#include <stdint.h>
void candidate_kernel(const int8_t *Q, const int8_t *K, int8_t *S,
                      int M, int N, int D,
                      int32_t scale_mult, int scale_shift) {
    for (int i = 0; i < M; i++) {
        for (int j = 0; j < N; j++) {
            int32_t acc = 0;
            for (int d = 0; d < D; d++)
                acc += (int32_t)Q[i*D+d] * (int32_t)K[d*N+j];
            /* BUG: truncating shift, no round-half-away-from-zero */
            int64_t r = (int64_t)acc * (int64_t)scale_mult;
            int64_t q = r >> scale_shift;
            if (q >  127) q =  127;
            if (q < -128) q = -128;
            S[i*N+j] = (int8_t)q;
        }
    }
}
