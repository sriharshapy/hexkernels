/* Baseline: naive scalar sum-of-absolute-differences. Speedup denominator:
 * correct but no vectorization, no VTCM staging -- pays full DDR latency and
 * scalar throughput on every element. */
#include <stdint.h>
void candidate_kernel(const uint8_t *a, const uint8_t *b, int n, int32_t *out) {
    int32_t s = 0;
    for (int i = 0; i < n; i++) { int dd = (int)a[i] - (int)b[i]; s += dd < 0 ? -dd : dd; }
    out[0] = s;
}
