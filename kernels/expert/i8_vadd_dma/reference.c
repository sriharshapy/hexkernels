/* Baseline: pure scalar int8 elementwise add with two's-complement wraparound.
 * This is the speedup denominator; correct but unvectorized. */
#include <stdint.h>
#include "kernel_api.h"

void candidate_kernel(const int8_t *a, const int8_t *b, int8_t *out, int n) {
    for (int i = 0; i < n; i++) out[i] = (int8_t)(a[i] + b[i]);
}
