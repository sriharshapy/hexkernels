/* Near-miss: hardcodes gamma=1 and beta=0 — ignores runtime gamma/beta params.
 * Fails on any parameter set where gamma != 1 or beta != 0. */
#include <stdint.h>
void candidate_kernel(const int8_t *x, const int8_t *residual, int8_t *out, int n,
                      const int8_t *gamma, const int8_t *beta,
                      const uint8_t *inv_lut) {
    (void)gamma;   /* BUG: gamma ignored */
    (void)beta;    /* BUG: beta ignored */

    int8_t t[256];
    for (int i = 0; i < n; i++) {
        int32_t s = (int32_t)x[i] + (int32_t)residual[i];
        if (s >  127) s =  127;
        if (s < -128) s = -128;
        t[i] = (int8_t)s;
    }

    int32_t sum = 0;
    for (int i = 0; i < n; i++) sum += (int32_t)t[i];
    int32_t mu = sum / n;

    int32_t var_sum = 0;
    for (int i = 0; i < n; i++) {
        int32_t d = (int32_t)t[i] - mu;
        var_sum += d * d;
    }
    int32_t var = var_sum / n;
    int32_t v_idx = var < 0 ? 0 : (var > 255 ? 255 : var);
    uint8_t inv = inv_lut[(int)v_idx];

    for (int i = 0; i < n; i++) {
        int32_t d      = (int32_t)t[i] - mu;
        /* BUG: uses hardcoded gamma=1 (scaled = (d*1 + 64)>>7) and beta=0 */
        int32_t scaled = (d + 64) >> 7;
        int32_t normed = (scaled * (int32_t)inv + 128) >> 8;
        /* BUG: no beta add */
        if (normed >  127) normed =  127;
        if (normed < -128) normed = -128;
        out[i] = (int8_t)normed;
    }
}
