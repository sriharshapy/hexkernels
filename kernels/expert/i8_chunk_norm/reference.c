#include <stdint.h>
/*
 * Per-chunk normalization baseline -- pure integer, no FP.
 *
 * For each chunk c in [0, G):
 *  1. mu_c   = sum(xc[j]) / chunk
 *  2. var_c  = sum((xc[j]-mu_c)^2) / chunk
 *  3. v_idx  = clamp(var_c, 0, 255)
 *  4. inv_c  = inv_lut[v_idx]
 *  5. out[j] = clamp( ((xc[j]-mu_c)*gamma[j]+64)>>7 * inv_c+128)>>8 + beta[j], -128, 127 )
 */
void candidate_kernel(const int8_t *x, int8_t *out, int n, int G,
                      const int8_t *gamma, const int8_t *beta,
                      const uint8_t *inv_lut) {
    int chunk = n / G;
    for (int c = 0; c < G; c++) {
        const int8_t *xc  = x     + c * chunk;
        int8_t       *oc  = out   + c * chunk;
        const int8_t *gc  = gamma + c * chunk;
        const int8_t *bc  = beta  + c * chunk;

        /* Step 1: mean */
        int32_t sum = 0;
        for (int j = 0; j < chunk; j++) sum += (int32_t)xc[j];
        int32_t mu = sum / chunk;

        /* Step 2: variance */
        int32_t var_sum = 0;
        for (int j = 0; j < chunk; j++) {
            int32_t d = (int32_t)xc[j] - mu;
            var_sum += d * d;
        }
        int32_t var = var_sum / chunk;

        /* Step 3-4: inv LUT lookup */
        int32_t v_idx = var;
        if (v_idx < 0)   v_idx = 0;
        if (v_idx > 255) v_idx = 255;
        uint8_t inv = inv_lut[(int)v_idx];

        /* Step 5: per-element */
        for (int j = 0; j < chunk; j++) {
            int32_t d      = (int32_t)xc[j] - mu;
            int32_t scaled = (d * (int32_t)gc[j] + 64) >> 7;
            int32_t normed = (scaled * (int32_t)inv + 128) >> 8;
            int32_t res    = normed + (int32_t)bc[j];
            if (res >  127) res =  127;
            if (res < -128) res = -128;
            oc[j] = (int8_t)res;
        }
    }
}
