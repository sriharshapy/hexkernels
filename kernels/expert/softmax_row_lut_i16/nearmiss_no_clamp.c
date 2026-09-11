#include "kernel_api.h"
#include <stdint.h>

/* Plausible-but-WRONG: skips the clamp(diff, -255, 0) step before the +255
 * shift. For rows where the max-relative diff underflows past -255 (which
 * int16's wide range makes routine -- and the harness explicitly injects
 * such rows), this indexes exp_lut with a wildly out-of-[0,255] value,
 * reading garbage/wrong table entries and diverging from the reference. */
void candidate_kernel(const int16_t *x, int16_t *out, int R, int C,
                      const uint16_t *exp_lut) {
    for (int r = 0; r < R; r++) {
        const int16_t *row  = x   + (long)r * C;
        int16_t       *orow = out + (long)r * C;

        int16_t m = row[0];
        for (int j = 1; j < C; j++) if (row[j] > m) m = row[j];

        int32_t S = 0;
        for (int j = 0; j < C; j++) {
            int32_t diff = (int32_t)row[j] - (int32_t)m;
            /* BUG: no clamp to -255 here. */
            int idx = (int)diff + 255;
            uint16_t e = exp_lut[idx];
            S += (int32_t)e;
        }

        int32_t half_S = S / 2;
        for (int j = 0; j < C; j++) {
            int32_t diff = (int32_t)row[j] - (int32_t)m;
            int idx = (int)diff + 255;
            uint16_t e = exp_lut[idx];
            int64_t num = (int64_t)e * 32767 + half_S;
            orow[j] = (int16_t)(num / S);
        }
    }
}
