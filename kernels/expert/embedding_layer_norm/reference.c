#include <stdint.h>
/*
 * Embedding gather + LayerNorm baseline — pure integer, no FP.
 *
 * For each token i in [0, T):
 *   1. Gather: row = table[idx[i], :]
 *   2. LayerNorm on row (length D):
 *      a. mu    = sum(row[j]) / D                   (trunc toward zero)
 *      b. var   = sum((row[j]-mu)^2) / D            (trunc, >= 0)
 *      c. v_idx = clamp(var, 0, 255)
 *      d. inv   = inv_lut[v_idx]
 *      e. d[j]  = row[j] - mu
 *      f. scaled= (d[j]*gamma[j] + 64) >> 7         (round-half-up)
 *      g. normed= (scaled*inv + 128) >> 8           (round-half-up, SHIFT=8)
 *      h. out[i*D+j] = clamp(normed + beta[j], -128, 127)
 *
 * T=32 tokens, D=64 embedding dim, VOCAB=256.
 */
void candidate_kernel(const int8_t *table, const int32_t *idx, int8_t *out,
                      int T, int D,
                      const int8_t *gamma, const int8_t *beta,
                      const uint8_t *inv_lut) {
    for (int i = 0; i < T; i++) {
        const int8_t *row = table + (int)idx[i] * D;
        int8_t *out_row   = out + i * D;

        /* Step 2a: mean */
        int32_t sum = 0;
        for (int j = 0; j < D; j++) sum += (int32_t)row[j];
        int32_t mu = sum / D;

        /* Step 2b: variance */
        int32_t var_sum = 0;
        for (int j = 0; j < D; j++) {
            int32_t dv = (int32_t)row[j] - mu;
            var_sum += dv * dv;
        }
        int32_t var = var_sum / D;

        /* Step 2c-d: LUT lookup */
        int32_t v_idx = var;
        if (v_idx < 0)   v_idx = 0;
        if (v_idx > 255) v_idx = 255;
        uint8_t inv = inv_lut[(int)v_idx];

        /* Step 2e-h: per-dim normalize + affine */
        for (int j = 0; j < D; j++) {
            int32_t dv     = (int32_t)row[j] - mu;
            int32_t scaled = (dv * (int32_t)gamma[j] + 64) >> 7;
            int32_t normed = (scaled * (int32_t)inv + 128) >> 8;
            int32_t r      = normed + (int32_t)beta[j];
            if (r >  127) r =  127;
            if (r < -128) r = -128;
            out_row[j] = (int8_t)r;
        }
    }
}
