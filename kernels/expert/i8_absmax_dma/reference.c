/* Baseline: scalar int8 absolute-max reduction over DDR (no VTCM staging, no
 * vectorization). Widens to int32 so |-128|=128 fits without saturation.
 * Speedup denominator. */
#include <stdint.h>

void candidate_kernel(const int8_t *a, int n, int32_t *out) {
    int32_t m = 0;
    for (int i = 0; i < n; i++) {
        int32_t v = a[i] < 0 ? -(int32_t)a[i] : (int32_t)a[i];
        if (v > m) m = v;
    }
    out[0] = m;
}
