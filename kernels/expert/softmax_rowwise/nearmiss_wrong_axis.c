/* Near-miss: computes softmax over columns (axis=0) instead of rows (axis=1).
 * Iterates over each COLUMN j, finding max over all rows for that column.
 * Produces a different normalization — wrong for row-wise softmax. */
#include <stdint.h>
void candidate_kernel(const int8_t *x, uint8_t *out, int R, int C,
                      const uint8_t *exp_lut) {
    /* BUG: softmax over column axis instead of row axis */
    for (int j = 0; j < C; j++) {
        int8_t m = x[0*C + j];
        for (int r = 1; r < R; r++) if (x[r*C+j] > m) m = x[r*C+j];
        int32_t S = 0;
        for (int r = 0; r < R; r++) {
            int diff = (int)x[r*C+j] - (int)m;
            if (diff < -255) diff = -255;
            uint8_t e = exp_lut[diff + 255];
            out[r*C+j] = e;
            S += (int32_t)e;
        }
        int32_t half_S = S / 2;
        for (int r = 0; r < R; r++) {
            int32_t ej = (int32_t)out[r*C+j];
            out[r*C+j] = (uint8_t)((ej * 255 + half_S) / S);
        }
    }
}
