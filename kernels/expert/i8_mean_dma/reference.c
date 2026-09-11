/* Baseline: pure scalar int8 mean-reduction over DDR (no HVX, no VTCM
 * staging). Exact int32 scalar sum then integer mean. Speedup denominator. */
#include <stdint.h>
#include "kernel_api.h"

void candidate_kernel(const int8_t *a, int n, int32_t *out) {
    int32_t s = 0;
    for (int i = 0; i < n; i++) s += (int32_t)a[i];
    out[0] = s / n;
}
