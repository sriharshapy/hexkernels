#include <stdint.h>
/*
 * GQA group-reduce baseline: sum HPG=H/G head planes per group, then
 * requantize (round-half-away-from-zero) to int8.
 */
void candidate_kernel(const int8_t *X, int8_t *Y, int H, int G, int M, int N,
                      int32_t scale_mult, int scale_shift) {
    int MN  = M * N;
    int HPG = H / G;

    for (int g = 0; g < G; g++) {
        for (int idx = 0; idx < MN; idx++) {
            int32_t acc = 0;
            for (int t = 0; t < HPG; t++)
                acc += (int32_t)X[(long)(g*HPG + t) * MN + idx];

            int64_t r    = (int64_t)acc * (int64_t)scale_mult;
            int64_t half = (scale_shift > 0) ? ((int64_t)1 << (scale_shift - 1)) : 0;
            int64_t q    = (r >= 0) ? ((r + half) >> scale_shift)
                                    : -((-r + half) >> scale_shift);
            if (q >  127) q =  127;
            if (q < -128) q = -128;
            Y[(long)g * MN + idx] = (int8_t)q;
        }
    }
}
