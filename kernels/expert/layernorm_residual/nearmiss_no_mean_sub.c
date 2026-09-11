/* Near-miss: performs residual add correctly but omits mean subtraction in the
 * LayerNorm step — uses t[i] directly instead of (t[i]-mu).
 * Also computes variance without centering (E[x^2] instead of Var[x]). */
#include <stdint.h>
void candidate_kernel(const int8_t *x, const int8_t *residual, int8_t *out, int n,
                      const int8_t *gamma, const int8_t *beta,
                      const uint8_t *inv_lut) {
    int8_t t[256];
    for (int i = 0; i < n; i++) {
        int32_t s = (int32_t)x[i] + (int32_t)residual[i];
        if (s >  127) s =  127;
        if (s < -128) s = -128;
        t[i] = (int8_t)s;
    }

    /* BUG: no mean subtraction; var uses raw t[i]^2 */
    int32_t var_sum = 0;
    for (int i = 0; i < n; i++) var_sum += (int32_t)t[i] * (int32_t)t[i];
    int32_t var = var_sum / n;
    int32_t v_idx = var < 0 ? 0 : (var > 255 ? 255 : var);
    uint8_t inv = inv_lut[(int)v_idx];

    for (int i = 0; i < n; i++) {
        int32_t d      = (int32_t)t[i];   /* BUG: should be t[i]-mu */
        int32_t scaled = (d * (int32_t)gamma[i] + 64) >> 7;
        int32_t normed = (scaled * (int32_t)inv + 128) >> 8;
        int32_t r      = normed + (int32_t)beta[i];
        if (r >  127) r =  127;
        if (r < -128) r = -128;
        out[i] = (int8_t)r;
    }
}
