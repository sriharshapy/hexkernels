/* Near-miss: normalizes over the ENTIRE vector (global mean/var) instead of
 * per-chunk.  Semantically wrong for chunk_norm: uses one shared mu/var for all
 * G chunks rather than computing separate statistics per chunk. */
#include <stdint.h>
void candidate_kernel(const int8_t *x, int8_t *out, int n, int G,
                      const int8_t *gamma, const int8_t *beta,
                      const uint8_t *inv_lut) {
    /* BUG: global mean and variance instead of per-chunk */
    int32_t sum = 0;
    for (int i = 0; i < n; i++) sum += (int32_t)x[i];
    int32_t mu = sum / n;

    int32_t var_sum = 0;
    for (int i = 0; i < n; i++) {
        int32_t d = (int32_t)x[i] - mu;
        var_sum += d * d;
    }
    int32_t var = var_sum / n;

    int32_t v_idx = var;
    if (v_idx < 0)   v_idx = 0;
    if (v_idx > 255) v_idx = 255;
    uint8_t inv = inv_lut[(int)v_idx];

    for (int i = 0; i < n; i++) {
        int32_t d      = (int32_t)x[i] - mu;
        int32_t scaled = (d * (int32_t)gamma[i] + 64) >> 7;
        int32_t normed = (scaled * (int32_t)inv + 128) >> 8;
        int32_t res    = normed + (int32_t)beta[i];
        if (res >  127) res =  127;
        if (res < -128) res = -128;
        out[i] = (int8_t)res;
    }
}
