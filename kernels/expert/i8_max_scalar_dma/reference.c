/* Baseline: pure scalar max-with-scalar (no HVX, no DMA), reading/writing DDR
 * directly. Correct but unvectorized. Speedup denominator. */
#include <stdint.h>
#include "kernel_api.h"

void candidate_kernel(const int8_t *in, int8_t *out, int n, int8_t c) {
    for (int i = 0; i < n; i++) {
        int8_t x = in[i];
        out[i] = (x > c) ? x : c;
    }
}
