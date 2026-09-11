/* Baseline: scalar sum reduction reading a[] directly from DDR (no HVX, no
 * VTCM staging). Widens to int64 accumulator. Speedup denominator. */
#include <stdint.h>

void candidate_kernel(const int32_t *a, int n, int64_t *out) {
    int64_t s = 0;
    for (int i = 0; i < n; i++) s += (int64_t)a[i];
    out[0] = s;
}
