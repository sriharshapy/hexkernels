/* Baseline: naive scalar "count >= t" reduction. Speedup denominator: correct
 * but no vectorization, no VTCM staging -- pays full DDR latency and scalar
 * throughput on every element. */
#include <stdint.h>
void candidate_kernel(const uint8_t *a, int n, uint8_t t, int32_t *out) {
    int32_t s = 0;
    for (int i = 0; i < n; i++) if (a[i] >= t) s++;
    out[0] = s;
}
