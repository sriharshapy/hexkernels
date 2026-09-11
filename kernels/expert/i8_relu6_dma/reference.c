/* Baseline: scalar ReLU6 reading/writing DDR directly (no VTCM staging, no
 * HVX). relu6 = clamp(x, 0, cap), cap = 6*scale. Speedup denominator. */
#include <stdint.h>

void candidate_kernel(const int8_t *x, int8_t *out, int n, int8_t scale) {
    int8_t cap = (int8_t)(6 * (int)scale);
    for (int i = 0; i < n; i++) {
        int8_t v = x[i];
        if (v < 0) v = 0; else if (v > cap) v = cap;
        out[i] = v;
    }
}
