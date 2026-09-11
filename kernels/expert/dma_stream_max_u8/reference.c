/* Baseline: scalar unsigned-max reduction reading a[] directly from DDR (no
 * HVX, no VTCM staging). This is the speedup denominator. */
#include <stdint.h>

void candidate_kernel(const uint8_t *a, int n, uint8_t *out) {
    uint8_t m = 0;
    for (int i = 0; i < n; i++) if (a[i] > m) m = a[i];
    out[0] = m;
}
