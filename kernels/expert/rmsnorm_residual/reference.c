#include <stdint.h>
/*
 * RMSNorm-with-residual baseline — pure integer, no FP.
 *
 * Fused: out = RMSNorm(x + residual)
 * NO mean subtraction (RMSNorm definition). NO beta.
 *
 * Steps (pinned, SHIFT=8):
 *  1. t[i]   = clamp(x[i] + residual[i], -128, 127)   (saturating add)
 *  2. rms2   = sum(t[i]^2) / n                         (trunc, >= 0)
 *  3. r_idx  = clamp(rms2, 0, 255)
 *  4. inv    = inv_lut[r_idx]
 *  5a. scaled = (t[i]*gamma[i] + 64) >> 7              (round-half-up)
 *  5b. normed = (scaled*inv + 128) >> 8                (round-half-up, SHIFT=8)
 *  5c. out[i] = clamp(normed, -128, 127)
 */
void candidate_kernel(const int8_t *x, const int8_t *residual, int8_t *out, int n,
                      const int8_t *gamma, const uint8_t *inv_lut) {
    /* Step 1: saturating residual add */
    int8_t t[256];  /* n <= 256 */
    for (int i = 0; i < n; i++) {
        int32_t s = (int32_t)x[i] + (int32_t)residual[i];
        if (s >  127) s =  127;
        if (s < -128) s = -128;
        t[i] = (int8_t)s;
    }

    /* Step 2: rms2 */
    int32_t sum_sq = 0;
    for (int i = 0; i < n; i++) {
        int32_t ti = (int32_t)t[i];
        sum_sq += ti * ti;
    }
    int32_t rms2 = sum_sq / n;

    /* Step 3-4: LUT lookup */
    int32_t r_idx = rms2;
    if (r_idx < 0)   r_idx = 0;
    if (r_idx > 255) r_idx = 255;
    uint8_t inv = inv_lut[(int)r_idx];

    /* Step 5: per-element scale */
    for (int i = 0; i < n; i++) {
        int32_t ti     = (int32_t)t[i];
        int32_t scaled = (ti * (int32_t)gamma[i] + 64) >> 7;
        int32_t normed = (scaled * (int32_t)inv + 128) >> 8;
        if (normed >  127) normed =  127;
        if (normed < -128) normed = -128;
        out[i] = (int8_t)normed;
    }
}
