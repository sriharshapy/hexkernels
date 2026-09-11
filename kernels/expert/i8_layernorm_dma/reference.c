/* Baseline: competent per-row integer LayerNorm reading/writing DDR directly
 * (no VTCM staging). Each row's mean/variance/LUT/affine is computed with a
 * plain scalar loop (W=128 is small; the point of this task is the DMA/VTCM
 * row-block streaming, not the per-row math). Speedup denominator. */
#include <stdint.h>
#include <stddef.h>

static void layernorm_row(const int8_t *xr, int8_t *outr, int W,
                          const int8_t *gamma, const int8_t *beta,
                          const uint8_t *inv_lut) {
    int32_t sum = 0;
    for (int i = 0; i < W; i++) sum += (int32_t)xr[i];
    int32_t mu = sum / W;

    int32_t var_sum = 0;
    for (int i = 0; i < W; i++) {
        int32_t d = (int32_t)xr[i] - mu;
        var_sum += d * d;
    }
    int32_t var = var_sum / W;
    int32_t v_idx = var;
    if (v_idx < 0) v_idx = 0;
    if (v_idx > 255) v_idx = 255;
    uint8_t inv = inv_lut[(int)v_idx];

    for (int i = 0; i < W; i++) {
        int32_t d      = (int32_t)xr[i] - mu;
        int32_t scaled = (d * (int32_t)gamma[i] + 64) >> 7;
        int32_t normed = (scaled * (int32_t)inv + 128) >> 8;
        int32_t r      = normed + (int32_t)beta[i];
        if (r > 127) r = 127;
        if (r < -128) r = -128;
        outr[i] = (int8_t)r;
    }
}

void candidate_kernel(const int8_t *x, int8_t *out, int R, int W,
                      const int8_t *gamma, const int8_t *beta,
                      const uint8_t *inv_lut) {
    for (int r = 0; r < R; r++)
        layernorm_row(x + (size_t)r * W, out + (size_t)r * W, W, gamma, beta, inv_lut);
}
