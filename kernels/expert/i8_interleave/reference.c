/* Baseline: pure scalar 2-way interleave (zip), no HVX/DMA staging.
 * out[2*i] = a[i]; out[2*i+1] = b[i]. Speedup denominator. */
#include <stdint.h>
#include "kernel_api.h"

void candidate_kernel(const int8_t *a, const int8_t *b, int8_t *out, int n) {
    for (int i = 0; i < n; i++) {
        out[2*i]   = a[i];
        out[2*i+1] = b[i];
    }
}
