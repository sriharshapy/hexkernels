/* Near-miss: adds mean subtraction like LayerNorm — wrong for RMSNorm.
 * RMSNorm normalizes by RMS, NOT by (x-mean)/std. */
#include <stdint.h>
void candidate_kernel(const int8_t *x, const int8_t *residual, int8_t *out, int n,
                      const int8_t *gamma, const uint8_t *inv_lut) {
    int8_t t[256];
    for (int i = 0; i < n; i++) {
        int32_t s = (int32_t)x[i] + (int32_t)residual[i];
        if (s >  127) s =  127;
        if (s < -128) s = -128;
        t[i] = (int8_t)s;
    }

    /* BUG: subtracts mean before computing rms2 — changes the statistic */
    int32_t sum = 0;
    for (int i = 0; i < n; i++) sum += (int32_t)t[i];
    int32_t mu = sum / n;

    int32_t sum_sq = 0;
    for (int i = 0; i < n; i++) {
        int32_t d = (int32_t)t[i] - mu;  /* BUG: RMSNorm should use t[i], not (t[i]-mu) */
        sum_sq += d * d;
    }
    int32_t rms2 = sum_sq / n;
    int32_t r_idx = rms2 < 0 ? 0 : (rms2 > 255 ? 255 : rms2);
    uint8_t inv = inv_lut[(int)r_idx];

    for (int i = 0; i < n; i++) {
        int32_t ti     = (int32_t)t[i] - mu;  /* BUG: should be t[i], no mean sub */
        int32_t scaled = (ti * (int32_t)gamma[i] + 64) >> 7;
        int32_t normed = (scaled * (int32_t)inv + 128) >> 8;
        if (normed >  127) normed =  127;
        if (normed < -128) normed = -128;
        out[i] = (int8_t)normed;
    }
}
