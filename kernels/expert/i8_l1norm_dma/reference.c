/* Baseline: pure scalar L1-norm reduction (no HVX/DMA staging).
 * out[0] = sum_i |a[i]| (|-128|=128, int32 accumulator). Speedup denominator. */
#include <stdint.h>
#include "kernel_api.h"

void candidate_kernel(const int8_t *a, int n, int32_t *out) {
    int32_t s = 0;
    for (int i = 0; i < n; i++) { int32_t v = a[i] < 0 ? -(int32_t)a[i] : (int32_t)a[i]; s += v; }
    out[0] = s;
}
