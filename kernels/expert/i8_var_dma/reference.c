/* Baseline: pure scalar int8 integer-variance reduction.
 * out[0] = (n*sum(a[i]^2) - (sum a[i])^2) / n, integer floor, int64 intermediates
 * to match the harness's exact reference computation. */
#include <stdint.h>
#include "kernel_api.h"

void candidate_kernel(const int8_t *a, int n, int32_t *out) {
    int64_t sumx = 0, sumsq = 0;
    for (int i = 0; i < n; i++) {
        int64_t v = (int64_t)a[i];
        sumx += v;
        sumsq += v * v;
    }
    out[0] = (int32_t)(((int64_t)n * sumsq - sumx * sumx) / (int64_t)n);
}
