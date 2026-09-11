#include <stdint.h>
/*
 * Instance Normalization baseline — pure integer, no FP.
 * Normalizes each channel's spatial map (H*W elements) independently.
 *
 * Layout: x[c * HW + s], H=16, W=16, C=16, HW=256, n=4096.
 *
 * Steps per channel c (pinned, SHIFT=8):
 *  1. mu    = sum_s x[c*HW+s] / HW                 (trunc toward zero)
 *  2. var   = sum_s (x[c*HW+s]-mu)^2 / HW          (trunc, >= 0)
 *  3. v_idx = clamp(var, 0, 255)
 *  4. inv   = inv_lut[c*256 + v_idx]
 *  5a. d      = x[c*HW+s] - mu
 *  5b. scaled = (d*gamma[c] + 64) >> 7              (round-half-up)
 *  5c. normed = (scaled*inv + 128) >> 8             (round-half-up, SHIFT=8)
 *  5d. out    = clamp(normed + beta[c], -128, 127)
 */
void candidate_kernel(const int8_t *x, int8_t *out,
                      int H, int W, int C,
                      const int8_t *gamma, const int8_t *beta,
                      const uint8_t *inv_lut) {
    int hw = H * W;
    for (int c = 0; c < C; c++) {
        const uint8_t *clut = inv_lut + c * 256;
        int base = c * hw;

        /* Step 1: mean */
        int32_t sum = 0;
        for (int s = 0; s < hw; s++) sum += (int32_t)x[base + s];
        int32_t mu = sum / hw;

        /* Step 2: variance */
        int32_t var_sum = 0;
        for (int s = 0; s < hw; s++) {
            int32_t d = (int32_t)x[base + s] - mu;
            var_sum += d * d;
        }
        int32_t var = var_sum / hw;

        /* Step 3-4: LUT lookup */
        int32_t v_idx = var;
        if (v_idx < 0)   v_idx = 0;
        if (v_idx > 255) v_idx = 255;
        uint8_t inv = clut[(int)v_idx];

        /* Step 5: per-element normalize + affine */
        for (int s = 0; s < hw; s++) {
            int pos    = base + s;
            int32_t d  = (int32_t)x[pos] - mu;
            int32_t sc = (d * (int32_t)gamma[c] + 64) >> 7;
            int32_t nm = (sc * (int32_t)inv + 128) >> 8;
            int32_t r  = nm + (int32_t)beta[c];
            if (r >  127) r =  127;
            if (r < -128) r = -128;
            out[pos] = (int8_t)r;
        }
    }
}
