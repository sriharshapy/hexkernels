#include <stdint.h>
/*
 * Attention-score rescale baseline (round-half-away-from-zero, int32 -> int16).
 * Purely elementwise: M*N total scalar iterations.
 */
void candidate_kernel(const int32_t *raw, int16_t *out, int M, int N,
                      int32_t scale_mult, int scale_shift) {
    int total = M * N;
    for (int i = 0; i < total; i++) {
        int64_t r    = (int64_t)raw[i] * (int64_t)scale_mult;
        int64_t half = (scale_shift > 0) ? ((int64_t)1 << (scale_shift - 1)) : 0;
        int64_t q    = (r >= 0) ? ((r + half) >> scale_shift)
                                : -((-r + half) >> scale_shift);
        if (q >  32767) q =  32767;
        if (q < -32768) q = -32768;
        out[i] = (int16_t)q;
    }
}
