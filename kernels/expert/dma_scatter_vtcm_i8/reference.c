/* Baseline: scalar scatter directly to DDR. Since idx is a permutation,
 * every position is written exactly once (order-independent), but each
 * write goes to a scattered DDR address -- this is the speedup
 * denominator. */
#include <stdint.h>
void candidate_kernel(const int8_t *values, const int32_t *idx, int32_t *out, int n) {
    for (int i = 0; i < n; i++) out[idx[i]] = (int32_t)values[i];
}
