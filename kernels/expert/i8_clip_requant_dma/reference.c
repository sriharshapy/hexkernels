/* Baseline: plain scalar int32->int8 requantize+clip over DDR (no VTCM
 * staging, no HVX). Round-half-away via sign-split; int8 saturate then clip
 * to [lo,hi]. Speedup denominator. */
#include <stdint.h>

static int8_t ref_scalar(int32_t ai, int32_t mult, int shift, int8_t zp, int8_t lo, int8_t hi) {
    int64_t v = (int64_t)ai * (int64_t)mult;
    int64_t half = (shift > 0) ? ((int64_t)1 << (shift - 1)) : 0;
    int64_t r = (v >= 0) ? ((v + half) >> shift) : -(((-v) + half) >> shift);
    r += zp;
    if (r > 127) r = 127;
    if (r < -128) r = -128;
    if (r > hi) r = hi;
    if (r < lo) r = lo;
    return (int8_t)r;
}

void candidate_kernel(const int32_t *a, int8_t *out, int n,
                      int32_t mult, int shift, int8_t zp, int8_t lo, int8_t hi) {
    for (int i = 0; i < n; i++) {
        out[i] = ref_scalar(a[i], mult, shift, zp, lo, hi);
    }
}
