/* Baseline: naive scalar uint8 global-average-pool. Speedup denominator:
 * correct but no vectorization, no VTCM staging -- pays full DDR latency and
 * scalar throughput on every element. */
#include <stdint.h>
void candidate_kernel(const uint8_t *a, int n, uint8_t *out) {
    int32_t s = 0;
    for (int i = 0; i < n; i++) s += (int32_t)a[i];
    out[0] = (uint8_t)(s / n);
}
