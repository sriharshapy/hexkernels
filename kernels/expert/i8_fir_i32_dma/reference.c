/* Baseline: pure scalar 1D FIR correlation (no HVX/DMA staging).
 * out[i] = sum_{j=0}^{ntaps-1} x[i+j]*taps[j]. Speedup denominator. */
#include <stdint.h>
#include "kernel_api.h"

void candidate_kernel(const int8_t *x, const int8_t *taps, int32_t *out, int n, int ntaps) {
    for (int i = 0; i < n; i++) {
        int32_t acc = 0;
        for (int j = 0; j < ntaps; j++) acc += (int32_t)x[i + j] * (int32_t)taps[j];
        out[i] = acc;
    }
}
