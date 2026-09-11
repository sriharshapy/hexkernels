/* Baseline: pure scalar global average pool (no HVX/DMA staging).
 * out[0] = (int8_t)(sum(a[0..n-1]) / n), int32 accumulator, C truncation.
 * Speedup denominator. */
#include <stdint.h>
#include "kernel_api.h"

void candidate_kernel(const int8_t *a, int n, int8_t *out) {
    int32_t s = 0;
    for (int i = 0; i < n; i++) s += (int32_t)a[i];
    out[0] = (int8_t)(s / n);
}
