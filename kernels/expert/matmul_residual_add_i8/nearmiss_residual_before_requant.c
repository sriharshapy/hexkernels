/* Near-miss: adds the residual BEFORE the requantize/shift step instead of
 * after (a very common, plausible bug -- treating the residual as part of
 * the pre-shift accumulator rather than as a post-requantize saturating
 * add). Compiles fine; wrong whenever scale_shift makes a difference
 * (i.e. whenever scale_shift > 0, which both swept param sets include at
 * least one of).
 *   BUG: q = round_half_away((acc + residual) * scale_mult, scale_shift)
 *   CORRECT: requantized = round_half_away(acc*scale_mult, scale_shift);
 *            out = clamp(requantized + residual)
 */
#include <stdint.h>
void candidate_kernel(const int8_t *A, const int8_t *B, const int8_t *residual,
                      int8_t *out, int M, int N, int K,
                      int32_t scale_mult, int scale_shift)
{
    for (int i = 0; i < M; i++) {
        for (int j = 0; j < N; j++) {
            int32_t acc = 0;
            for (int k = 0; k < K; k++)
                acc += (int32_t)A[i*K + k] * (int32_t)B[k*N + j];

            /* BUG: residual folded in BEFORE the shift/requantize. */
            int32_t acc_plus_res = acc + (int32_t)residual[i*N + j];

            int64_t r    = (int64_t)acc_plus_res * (int64_t)scale_mult;
            int64_t half = (scale_shift > 0) ? ((int64_t)1 << (scale_shift - 1)) : 0;
            int64_t q    = (r >= 0) ? ((r + half) >> scale_shift)
                                    : -((-r + half) >> scale_shift);
            if (q >  127) q =  127;
            if (q < -128) q = -128;
            out[i*N + j] = (int8_t)q;
        }
    }
}
