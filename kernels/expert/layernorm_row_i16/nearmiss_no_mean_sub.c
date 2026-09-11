/* Near-miss: omits mean subtraction in the elementwise epilogue -- uses
 * raw x[r][c] instead of (x[r][c]-mu) in step 5a (mu/var/inv are still
 * computed correctly and used for vidx/inv, but the affine step is never
 * centered). Compiles, plausible ("forgot to subtract mu before scaling"),
 * fails whenever a row's mu != 0 (true for all non-constant rows here). */
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

        for (int c = 0; c < C; c++) {
            int32_t d      = (int32_t)xr[c];   /* BUG: should be xr[c]-mu */
            int32_t scaled = (d * (int32_t)gamma[c] + 32) >> 6;
            int32_t normed = (scaled * inv + 512) >> 10;
            int32_t res    = normed + (int32_t)beta[c];
            if (res >  32767) res =  32767;
            if (res < -32768) res = -32768;
            outr[c] = (int16_t)res;
        }
    }
}
