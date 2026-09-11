/* NEAR-MISS: skips the recip-lut normalization step entirely and instead
 * scales each e_j by a fixed 255/C constant (a classic "forgot proper
 * normalization" bug -- S/sidx/recip_lut are never used at all). Compiles,
 * produces plausible-looking uint8 output, but does not match the pinned
 * reciprocal-LUT formula -- guaranteed wrong. */
#include <stdint.h>

void candidate_kernel(const int8_t *x, uint8_t *out, int R, int C,
                      const uint8_t *exp_lut, const uint16_t *recip_lut) {
    (void)recip_lut;
    int32_t scale = 255 / C;   /* WRONG: fixed constant instead of recip_lut */
    if (scale < 1) scale = 1;

    for (int r = 0; r < R; r++) {
        const int8_t *row = x + r * C;
        uint8_t      *orow = out + r * C;

        int8_t m = row[0];
        for (int j = 1; j < C; j++) if (row[j] > m) m = row[j];

        for (int j = 0; j < C; j++) {
            int diff = (int)row[j] - (int)m;
            if (diff < -255) diff = -255;
            if (diff > 0) diff = 0;
            int32_t ej = (int32_t)exp_lut[diff + 255];
            int32_t v = ej * scale;   /* WRONG: no recip-lut normalization */
            if (v < 0) v = 0;
            if (v > 255) v = 255;
            orow[j] = (uint8_t)v;
        }
    }
}
