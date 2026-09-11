/* NEAR-MISS: off-by-one -- sums only HPG-1 heads per group (drops the
 * last head-plane in each group) instead of all HPG. Compiles, stays
 * in-bounds, guaranteed wrong whenever the dropped head's values are
 * nonzero (true for essentially every element in the harness). */
#include "kernel_api.h"
#include <stdint.h>

static int8_t requant_i8(int32_t acc, int32_t mult, int shift) {
    int64_t r    = (int64_t)acc * (int64_t)mult;
    int64_t half = (shift > 0) ? ((int64_t)1 << (shift - 1)) : 0;
    int64_t q    = (r >= 0) ? ((r + half) >> shift) : -((-r + half) >> shift);
    if (q >  127) q =  127;
    if (q < -128) q = -128;
    return (int8_t)q;
}

void candidate_kernel(const int8_t *X, int8_t *Y, int H, int G, int M, int N,
                      int32_t scale_mult, int scale_shift) {
    int MN  = M * N;
    int HPG = H / G;

    for (int g = 0; g < G; g++) {
        for (int idx = 0; idx < MN; idx++) {
            int32_t acc = 0;
            for (int t = 0; t < HPG - 1; t++)   /* WRONG: drops the last head */
                acc += (int32_t)X[(long)(g*HPG + t) * MN + idx];
            Y[(long)g * MN + idx] = requant_i8(acc, scale_mult, scale_shift);
        }
    }
}
