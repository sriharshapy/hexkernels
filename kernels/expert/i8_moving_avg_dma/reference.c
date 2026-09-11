/* Baseline: pure scalar box moving average (W=8), reading/writing DDR
 * directly (no HVX, no VTCM staging). Truncate-toward-zero divide (C integer
 * /). Speedup denominator. */
#include <stdint.h>
#include "kernel_api.h"

void candidate_kernel(const int8_t *x, int8_t *out, int n, int W) {
    for (int i = 0; i < n; i++) {
        int32_t acc = 0;
        for (int j = 0; j < W; j++) acc += (int32_t)x[i + j];
        out[i] = (int8_t)(acc / W);
    }
}
