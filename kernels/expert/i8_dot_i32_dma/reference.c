/* Baseline: pure scalar int8 dot product (no HVX/DMA staging).
 * out[0] = sum_i a[i]*b[i], int32 accumulator. Speedup denominator. */
#include <stdint.h>
#include "kernel_api.h"
void candidate_kernel(const int8_t *a, const int8_t *b, int n, int32_t *out) {
    int32_t s = 0;
    for (int i = 0; i < n; i++) s += (int32_t)a[i] * (int32_t)b[i];
    out[0] = s;
}
