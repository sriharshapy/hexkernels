#include <stdint.h>
/*
 * Attention QKᵀ tile (column-major K) baseline.
 *
 * For each (i,j):
 *  1. raw  = sum_d Q[i*D+d] * K[d*N+j]               (int32; K column-major)
 *  2. r    = (int64_t)raw * scale_mult
 *  3. half = scale_shift > 0 ? (1LL << (scale_shift-1)) : 0
 *  4. q    = round-half-away: (r>=0) ? (r+half)>>ss : -((-r+half)>>ss)
 *  5. S[i*N+j] = clamp(q, -128, 127)
 */
void candidate_kernel(const int8_t *Q, const int8_t *K, int8_t *S,
                      int M, int N, int D,
                      int32_t scale_mult, int scale_shift) {
    for (int i = 0; i < M; i++) {
        for (int j = 0; j < N; j++) {
            int32_t acc = 0;
            for (int d = 0; d < D; d++) {
                acc += (int32_t)Q[i*D + d] * (int32_t)K[d*N + j];
            }
            int64_t r    = (int64_t)acc * (int64_t)scale_mult;
            int64_t half = (scale_shift > 0) ? ((int64_t)1 << (scale_shift - 1)) : 0;
            int64_t q    = (r >= 0) ? ((r + half) >> scale_shift)
                                    : -((-r + half) >> scale_shift);
            if (q >  127) q =  127;
            if (q < -128) q = -128;
            S[i*N + j] = (int8_t)q;
        }
    }
}
