/* Baseline: plain scalar signed greater-than mask reading/writing DDR directly
 * (no VTCM staging, no HVX). This is the speedup denominator. */
#include <stdint.h>
void candidate_kernel(const int8_t *a, const int8_t *b, uint8_t *out, int n) {
    for (int i = 0; i < n; i++) {
        out[i] = (uint8_t)(a[i] > b[i] ? 255 : 0);
    }
}
