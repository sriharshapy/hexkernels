/* Baseline: plain scalar saturating scalar-bias add over DDR (no VTCM staging,
 * no HVX). Valid while bias is within int8 range (signed saturating byte add).
 * Speedup denominator. */
#include <stdint.h>
void candidate_kernel(const int8_t *x, int8_t *out, int n, int32_t bias) {
    for (int i = 0; i < n; i++) {
        int32_t r = (int32_t)x[i] + bias;
        if (r > 127) r = 127;
        if (r < -128) r = -128;
        out[i] = (int8_t)r;
    }
}
