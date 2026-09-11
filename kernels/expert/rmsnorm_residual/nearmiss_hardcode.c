/* Near-miss: hardcodes gamma=1 — ignores runtime gamma param.
 * Fails on any parameter set where gamma != 1. */
#include <stdint.h>
void candidate_kernel(const int8_t *x, const int8_t *residual, int8_t *out, int n,
                      const int8_t *gamma, const uint8_t *inv_lut) {
    (void)gamma;  /* BUG: gamma ignored */

    int8_t t[256];
    for (int i = 0; i < n; i++) {
        int32_t s = (int32_t)x[i] + (int32_t)residual[i];
        if (s >  127) s =  127;
        if (s < -128) s = -128;
        t[i] = (int8_t)s;
    }

    int32_t sum_sq = 0;
    for (int i = 0; i < n; i++) {
        int32_t ti = (int32_t)t[i];
        sum_sq += ti * ti;
    }
    int32_t rms2 = sum_sq / n;
    int32_t r_idx = rms2 < 0 ? 0 : (rms2 > 255 ? 255 : rms2);
    uint8_t inv = inv_lut[(int)r_idx];

    for (int i = 0; i < n; i++) {
        int32_t ti     = (int32_t)t[i];
        /* BUG: gamma=1 hardcoded -> (ti*1 + 64) >> 7 */
        int32_t scaled = (ti + 64) >> 7;
        int32_t normed = (scaled * (int32_t)inv + 128) >> 8;
        if (normed >  127) normed =  127;
        if (normed < -128) normed = -128;
        out[i] = (int8_t)normed;
    }
}
