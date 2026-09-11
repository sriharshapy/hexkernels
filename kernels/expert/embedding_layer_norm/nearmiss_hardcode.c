/* Near-miss: hardcodes gamma=1 and beta=0 — ignores runtime gamma/beta.
 * Fails on any parameter set where gamma != 1 or beta != 0. */
#include <stdint.h>
void candidate_kernel(const int8_t *table, const int32_t *idx, int8_t *out,
                      int T, int D,
                      const int8_t *gamma, const int8_t *beta,
                      const uint8_t *inv_lut) {
    (void)gamma;  /* BUG: ignored */
    (void)beta;   /* BUG: ignored */
    for (int i = 0; i < T; i++) {
        const int8_t *row = table + (int)idx[i] * D;
        int8_t *out_row   = out + i * D;

        int32_t sum = 0;
        for (int j = 0; j < D; j++) sum += (int32_t)row[j];
        int32_t mu = sum / D;

        int32_t var_sum = 0;
        for (int j = 0; j < D; j++) {
            int32_t dv = (int32_t)row[j] - mu;
            var_sum += dv * dv;
        }
        int32_t var = var_sum / D;
        int32_t v_idx = var < 0 ? 0 : (var > 255 ? 255 : var);
        uint8_t inv = inv_lut[(int)v_idx];

        for (int j = 0; j < D; j++) {
            int32_t dv     = (int32_t)row[j] - mu;
            /* BUG: gamma=1, beta=0 hardcoded */
            int32_t scaled = (dv + 64) >> 7;
            int32_t normed = (scaled * (int32_t)inv + 128) >> 8;
            if (normed >  127) normed =  127;
            if (normed < -128) normed = -128;
            out_row[j] = (int8_t)normed;
        }
    }
}
