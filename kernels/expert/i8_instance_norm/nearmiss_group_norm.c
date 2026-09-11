/* Near-miss: confuses instance norm with group norm — normalizes ALL channels
 * together (one global statistics) instead of per-channel.
 * Produces wrong mean/variance since it pools across channels. */
#include <stdint.h>
void candidate_kernel(const int8_t *x, int8_t *out,
                      int H, int W, int C,
                      const int8_t *gamma, const int8_t *beta,
                      const uint8_t *inv_lut) {
    int hw = H * W;
    int n = C * hw;

    /* BUG: computes statistics globally across ALL channels */
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
    /* BUG: uses channel 0 LUT for all */
    uint8_t inv = inv_lut[(int)v_idx];

    for (int c = 0; c < C; c++) {
        for (int s = 0; s < hw; s++) {
            int pos    = c * hw + s;
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
