#include <stdint.h>
/*
 * LayerNorm-with-residual baseline — pure integer, no FP.
 *
 * Fused: out = LayerNorm(x + residual)
 *
 * Steps (pinned, SHIFT=8):
 *  1. t[i]   = clamp(x[i] + residual[i], -128, 127)   (saturating add)
 *  2. mu     = sum(t[i]) / n                           (trunc toward zero)
 *  3. var    = sum((t[i]-mu)^2) / n                    (trunc, >= 0)
 *  4. v_idx  = clamp(var, 0, 255)
 *  5. inv    = inv_lut[v_idx]
 *  6a. d[i]   = t[i] - mu
 *  6b. scaled = (d[i]*gamma[i] + 64) >> 7              (round-half-up)
 *  6c. normed = (scaled*inv + 128) >> 8                (round-half-up, SHIFT=8)
 *  6d. out[i] = clamp(normed + beta[i], -128, 127)
 */
void candidate_kernel(const int8_t *x, const int8_t *residual, int8_t *out, int n,
                      const int8_t *gamma, const int8_t *beta,
                      const uint8_t *inv_lut) {
    /* Step 1: saturating residual add */
    int8_t t[256];  /* n <= 256 */
    for (int i = 0; i < n; i++) {
        int32_t s = (int32_t)x[i] + (int32_t)residual[i];
        if (s >  127) s =  127;
        if (s < -128) s = -128;
        t[i] = (int8_t)s;
    }

    /* Step 2: mean */
    int32_t sum = 0;
    for (int i = 0; i < n; i++) sum += (int32_t)t[i];
    int32_t mu = sum / n;

    /* Step 3: variance */
    int32_t var_sum = 0;
    for (int i = 0; i < n; i++) {
        int32_t d = (int32_t)t[i] - mu;
        var_sum += d * d;
    }
    int32_t var = var_sum / n;

    /* Step 4-5: LUT lookup */
    int32_t v_idx = var;
    if (v_idx < 0)   v_idx = 0;
    if (v_idx > 255) v_idx = 255;
    uint8_t inv = inv_lut[(int)v_idx];

    /* Step 6: per-element normalize + affine */
    for (int i = 0; i < n; i++) {
        int32_t d      = (int32_t)t[i] - mu;
        int32_t scaled = (d * (int32_t)gamma[i] + 64) >> 7;
        int32_t normed = (scaled * (int32_t)inv + 128) >> 8;
        int32_t r      = normed + (int32_t)beta[i];
        if (r >  127) r =  127;
        if (r < -128) r = -128;
        out[i] = (int8_t)r;
    }
}
