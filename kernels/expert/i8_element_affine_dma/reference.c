/* Baseline: pure scalar affine-requantize (no HVX/DMA staging).
 * out[i] = sat8( round_half_away_from_zero(a[i]*scale + shift) >> s )
 * Speedup denominator. */
#include <stdint.h>
#include "kernel_api.h"

static int8_t ref_scalar(int8_t ai, int32_t scale, int32_t shift, int s) {
    int64_t v = (int64_t)ai * (int64_t)scale + (int64_t)shift;
    int64_t half = (s > 0) ? ((int64_t)1 << (s - 1)) : 0;
    int64_t r = (v >= 0) ? ((v + half) >> s) : -(((-v) + half) >> s);
    if (r >  127) r =  127;
    if (r < -128) r = -128;
    return (int8_t)r;
}

void candidate_kernel(const int8_t *a, int8_t *out, int n,
                      int32_t scale, int32_t shift, int s) {
    for (int i = 0; i < n; i++) out[i] = ref_scalar(a[i], scale, shift, s);
}
