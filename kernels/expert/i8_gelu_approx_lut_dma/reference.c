/* Baseline: pure scalar GELU-approx via 256-entry LUT with input requantisation
 * (no HVX/DMA staging).
 *   idx = clamp( (int)in[i] * scale / 64 + 128, 0, 255 )   ( / is C trunc toward 0 )
 *   out[i] = lut[idx]
 * Speedup denominator. */
#include <stdint.h>
#include "kernel_api.h"

static int clamp_idx(int x, int scale) {
    int idx = (x * scale) / 64 + 128; if (idx < 0) idx = 0; if (idx > 255) idx = 255; return idx;
}

void candidate_kernel(const int8_t *in, int8_t *out, int n, const int8_t *lut, int8_t scale) {
    for (int i = 0; i < n; i++) out[i] = lut[clamp_idx((int)in[i], (int)scale)];
}
