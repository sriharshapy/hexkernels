/* Baseline: pure scalar int8 multiply+requantize over DDR (no HVX, no VTCM
 * staging). Round-half-away-from-zero via int64 exact math, sat8 clamp.
 * Speedup denominator. */
#include <stdint.h>
#include "kernel_api.h"

void candidate_kernel(const int8_t *a, const int8_t *b, int8_t *out, int n,
                      int32_t mult, int shift, int8_t zp) {
    for (int i = 0; i < n; i++) {
        int32_t prod = (int32_t)a[i] * (int32_t)b[i];
        int64_t v = (int64_t)prod * (int64_t)mult;
        int64_t half = (shift > 0) ? ((int64_t)1 << (shift - 1)) : 0;
        int64_t r = (v >= 0) ? ((v + half) >> shift) : -(((-v) + half) >> shift);
        r += zp;
        if (r >  127) r =  127;
        if (r < -128) r = -128;
        out[i] = (int8_t)r;
    }
}
