/* Baseline: pure scalar leaky-ReLU (no HVX, no DMA). Correct but unvectorized;
 * reads/writes DDR directly. Speedup denominator.
 * Negative path: sat8((x*alpha)>>shift); positive path passes x through. */
#include <stdint.h>
#include "kernel_api.h"

void candidate_kernel(const int8_t *x, int8_t *out, int n, int alpha, int shift) {
    for (int i = 0; i < n; i++) {
        if (x[i] > 0) { out[i] = x[i]; continue; }
        int v = ((int)x[i] * alpha) >> shift;
        if (v > 127) v = 127;
        if (v < -128) v = -128;
        out[i] = (int8_t)v;
    }
}
