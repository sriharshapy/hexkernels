/* Near-miss: skips the residual add — applies LayerNorm to x directly,
 * ignoring the residual input. Semantically wrong for the fused op. */
#include <stdint.h>
void candidate_kernel(const int8_t *x, const int8_t *residual, int8_t *out, int n,
                      const int8_t *gamma, const int8_t *beta,
                      const uint8_t *inv_lut) {
    (void)residual;  /* BUG: residual ignored */

    int32_t sum = 0;
    for (int i = 0; i < n; i++) sum += (int32_t)x[i];
    int32_t mu = sum / n;

    int32_t var_sum = 0;
    for (int i = 0; i < n; i++) {
        int32_t d = (int32_t)x[i] - mu;
        var_sum += d * d;
    }
    int32_t var = var_sum / n;
    int32_t v_idx = var < 0 ? 0 : (var > 255 ? 255 : var);
    uint8_t inv = inv_lut[(int)v_idx];

    for (int i = 0; i < n; i++) {
        int32_t d      = (int32_t)x[i] - mu;   /* BUG: should use t[i]=x[i]+residual[i] */
        int32_t scaled = (d * (int32_t)gamma[i] + 64) >> 7;
        int32_t normed = (scaled * (int32_t)inv + 128) >> 8;
        int32_t r      = normed + (int32_t)beta[i];
        if (r >  127) r =  127;
        if (r < -128) r = -128;
        out[i] = (int8_t)r;
    }
}
