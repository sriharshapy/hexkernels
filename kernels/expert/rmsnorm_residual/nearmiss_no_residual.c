/* Near-miss: skips the residual add — applies RMSNorm to x directly. */
#include <stdint.h>
void candidate_kernel(const int8_t *x, const int8_t *residual, int8_t *out, int n,
                      const int8_t *gamma, const uint8_t *inv_lut) {
    (void)residual;  /* BUG: residual ignored */

    int32_t sum_sq = 0;
    for (int i = 0; i < n; i++) {
        int32_t xi = (int32_t)x[i];
        sum_sq += xi * xi;
    }
    int32_t rms2 = sum_sq / n;
    int32_t r_idx = rms2 < 0 ? 0 : (rms2 > 255 ? 255 : rms2);
    uint8_t inv = inv_lut[(int)r_idx];

    for (int i = 0; i < n; i++) {
        int32_t xi     = (int32_t)x[i];  /* BUG: should use t[i]=x[i]+residual[i] */
        int32_t scaled = (xi * (int32_t)gamma[i] + 64) >> 7;
        int32_t normed = (scaled * (int32_t)inv + 128) >> 8;
        if (normed >  127) normed =  127;
        if (normed < -128) normed = -128;
        out[i] = (int8_t)normed;
    }
}
