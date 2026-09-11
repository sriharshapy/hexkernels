/* Baseline: pure scalar global max pool (signed max, no HVX/DMA staging).
 * out[0] = max(a[0], ..., a[n-1]). Speedup denominator. */
#include <stdint.h>
#include "kernel_api.h"

void candidate_kernel(const int8_t *a, int n, int8_t *out) {
    int8_t m = -128;
    for (int i = 0; i < n; i++) if (a[i] > m) m = a[i];
    out[0] = m;
}
