/* Baseline: plain scalar argmin over DDR (no VTCM staging, no HVX).
 * out[0] = index of the first occurrence of the minimum value (strict <).
 * Speedup denominator. */
#include <stdint.h>

void candidate_kernel(const int8_t *a, int n, int32_t *out) {
    int bi = 0;
    int bv = (int)a[0];
    for (int i = 1; i < n; i++) {
        if ((int)a[i] < bv) { bv = a[i]; bi = i; }
    }
    out[0] = bi;
}
