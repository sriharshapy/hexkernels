/* Baseline: pure scalar int8 hard-swish (no HVX/DMA staging).
 *   relu6 = clamp(x+3, 0, 6)
 *   out   = clamp((x * relu6) / 6, -128, 127)   (C truncating division)
 * Speedup denominator. */
#include <stdint.h>
#include "kernel_api.h"

void candidate_kernel(const int8_t *x, int8_t *out, int n) {
    for (int i = 0; i < n; i++) {
        int v = (int)x[i];
        int r6 = v + 3; if (r6 < 0) r6 = 0; else if (r6 > 6) r6 = 6;
        int prod = v * r6;
        int result = prod / 6;              /* C truncation toward zero */
        if (result > 127) result = 127;
        if (result < -128) result = -128;
        out[i] = (int8_t)result;
    }
}
