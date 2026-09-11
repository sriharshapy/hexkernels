/* Near-miss: computes Q*K^T correctly but forgets to apply the scale requantization.
 * Clamps the raw int32 accumulator directly to int8, which is almost always all -128/+127
 * for realistic D=130 (accumulator range >> int8 range without scaling). */
#include <stdint.h>
void candidate_kernel(const int8_t *Q, const int8_t *K, int8_t *S,
                      int M, int N, int D,
                      int32_t scale_mult, int scale_shift) {
    /* BUG: no scale — raw accumulator clamped directly */
    (void)scale_mult; (void)scale_shift;
    for (int i = 0; i < M; i++) {
        for (int j = 0; j < N; j++) {
            int32_t acc = 0;
            for (int d = 0; d < D; d++)
                acc += (int32_t)Q[i*D+d] * (int32_t)K[j*D+d];
            if (acc >  127) acc =  127;
            if (acc < -128) acc = -128;
            S[i*N+j] = (int8_t)acc;
        }
    }
}
