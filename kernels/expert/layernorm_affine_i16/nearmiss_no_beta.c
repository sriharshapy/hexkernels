/* NEAR-MISS: correctly computes mu/var/inv/normed/scaled but omits the
 * final "+ beta[r]" add (a clean, deterministic, in-bounds bug -- beta is
 * read but never applied). Compiles, stays in-bounds, guaranteed wrong
 * whenever beta[r] != 0 (rows 0,1,2,3,5 in the harness all have nonzero
 * beta). */
#include "kernel_api.h"
#include <stdint.h>

void candidate_kernel(const int16_t *x, int16_t *out, int R, int C,
                      const int16_t *gamma, const int16_t *beta,
                      const uint16_t *inv_lut) {
    for (int r = 0; r < R; r++) {
        const int16_t *xr = x + (long)r * C;
        int16_t *outr = out + (long)r * C;

        int64_t sum = 0;
        for (int c = 0; c < C; c++) sum += (int64_t)xr[c];
        int32_t mu = (int32_t)(sum / C);

        int64_t var_sum = 0;
        for (int c = 0; c < C; c++) {
            int64_t d = (int64_t)xr[c] - mu;
            var_sum += d * d;
        }
        int64_t var = var_sum / C;
        int32_t vidx = (int32_t)(var >> 5);
        if (vidx < 0) vidx = 0;
        if (vidx > 255) vidx = 255;
        int32_t inv = (int32_t)inv_lut[vidx];

        int32_t g = (int32_t)gamma[r];
        (void)beta;  /* WRONG: beta never applied */

        for (int c = 0; c < C; c++) {
            int32_t d      = (int32_t)xr[c] - mu;
            int32_t normed = (d * inv + 512) >> 10;
            int32_t scaled = (normed * g + 32) >> 6;
            int32_t res    = scaled;   /* WRONG: missing "+ beta[r]" */
            if (res >  32767) res =  32767;
            if (res < -32768) res = -32768;
            outr[c] = (int16_t)res;
        }
    }
}
