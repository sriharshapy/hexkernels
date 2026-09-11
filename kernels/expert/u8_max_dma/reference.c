/* Baseline: naive scalar uint8 max-reduction. Speedup denominator: correct
 * but no vectorization, no VTCM staging -- pays full DDR latency and scalar
 * throughput on every element. */
#include <stdint.h>
void candidate_kernel(const uint8_t *a, int n, uint8_t *out) {
    uint8_t m = 0;
    for (int i = 0; i < n; i++) if (a[i] > m) m = a[i];
    out[0] = m;
}
