/* Baseline: naive scalar int32 sum-reduction. Speedup denominator: correct
 * but no vectorization, no VTCM staging -- pays full DDR latency and scalar
 * throughput on every element. */
#include <stdint.h>
void candidate_kernel(const int32_t *a, int n, int32_t *out) {
    int32_t s = 0;
    for (int i = 0; i < n; i++) s += a[i];
    out[0] = s;
}
