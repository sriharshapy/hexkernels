/* Baseline: straight scalar saturating abs reading/writing DDR directly (no
 * VTCM staging, no vectorization). This is the speedup denominator. */
#include <stdint.h>

void candidate_kernel(const int8_t *a, int8_t *out, int n) {
    for (int i = 0; i < n; i++) {
        int v = a[i];
        if (v < 0) v = -v;
        if (v > 127) v = 127;
        out[i] = (int8_t)v;
    }
}
