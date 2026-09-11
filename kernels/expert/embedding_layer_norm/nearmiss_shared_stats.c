/* Near-miss: computes LayerNorm statistics globally across all tokens
 * (one mu/var for the entire output) instead of per-token.
 * Produces wrong output for non-uniform token distributions. */
#include <stdint.h>
void candidate_kernel(const int8_t *table, const int32_t *idx, int8_t *out,
                      int T, int D,
                      const int8_t *gamma, const int8_t *beta,
                      const uint8_t *inv_lut) {
    /* BUG: gather ALL rows first into a temporary, then compute statistics globally */
    int8_t gathered[32 * 64];  /* T*D max */

    for (int i = 0; i < T; i++) {
        const int8_t *row = table + (int)idx[i] * D;
        for (int j = 0; j < D; j++) gathered[i * D + j] = row[j];
    }

    /* BUG: one global mean/variance across all T*D elements */
    int32_t sum = 0;
    for (int i = 0; i < T * D; i++) sum += (int32_t)gathered[i];
    int32_t mu = sum / (T * D);

    int32_t var_sum = 0;
    for (int i = 0; i < T * D; i++) {
        int32_t dv = (int32_t)gathered[i] - mu;
        var_sum += dv * dv;
    }
    int32_t var = var_sum / (T * D);
    int32_t v_idx = var < 0 ? 0 : (var > 255 ? 255 : var);
    uint8_t inv = inv_lut[(int)v_idx];

    for (int i = 0; i < T; i++) {
        for (int j = 0; j < D; j++) {
            int pos    = i * D + j;
            int32_t dv = (int32_t)gathered[pos] - mu;
            int32_t sc = (dv * (int32_t)gamma[j] + 64) >> 7;
            int32_t nm = (sc * (int32_t)inv + 128) >> 8;
            int32_t r  = nm + (int32_t)beta[j];
            if (r >  127) r =  127;
            if (r < -128) r = -128;
            out[pos] = (int8_t)r;
        }
    }
}
