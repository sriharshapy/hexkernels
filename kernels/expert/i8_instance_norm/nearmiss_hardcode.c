/* Near-miss: hardcodes gamma=1 and beta=0 — ignores runtime gamma/beta.
 * Fails on parameter sets where gamma != 1 or beta != 0. */
#include <stdint.h>
void candidate_kernel(const int8_t *x, int8_t *out,
                      int H, int W, int C,
                      const int8_t *gamma, const int8_t *beta,
                      const uint8_t *inv_lut) {
    (void)gamma;  /* BUG: ignored */
    (void)beta;   /* BUG: ignored */
    int hw = H * W;

    for (int c = 0; c < C; c++) {
        const uint8_t *clut = inv_lut + c * 256;
        int base = c * hw;

        int32_t sum = 0;
        for (int s = 0; s < hw; s++) sum += (int32_t)x[base + s];
        int32_t mu = sum / hw;

        int32_t var_sum = 0;
        for (int s = 0; s < hw; s++) {
            int32_t d = (int32_t)x[base + s] - mu;
            var_sum += d * d;
        }
        int32_t var = var_sum / hw;
        int32_t v_idx = var < 0 ? 0 : (var > 255 ? 255 : var);
        uint8_t inv = clut[(int)v_idx];

        for (int s = 0; s < hw; s++) {
            int pos    = base + s;
            int32_t d  = (int32_t)x[pos] - mu;
            /* BUG: gamma=1 hardcoded */
            int32_t sc = (d + 64) >> 7;
            int32_t nm = (sc * (int32_t)inv + 128) >> 8;
            /* BUG: beta=0 hardcoded */
            if (nm >  127) nm =  127;
            if (nm < -128) nm = -128;
            out[pos] = (int8_t)nm;
        }
    }
}
