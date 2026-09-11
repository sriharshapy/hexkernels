#include <stdint.h>
/*
 * Row-wise softmax baseline — pure integer, no FP.
 * For each row r in [0,R):
 *   1. m_r    = max over j in [0,C)
 *   2. idx_j  = clamp(x[r*C+j] - m_r, -255, 0) + 255
 *   3. e_j    = exp_lut[idx_j]
 *   4. S_r    = sum of e_j
 *   5. out[r*C+j] = (uint8)((e_j * 255 + S_r/2) / S_r)
 */
void candidate_kernel(const int8_t *x, uint8_t *out, int R, int C,
                      const uint8_t *exp_lut) {
    for (int r = 0; r < R; r++) {
        const int8_t *row = x + r * C;
        uint8_t      *orow = out + r * C;

        /* Step 1: row max */
        int8_t m = row[0];
        for (int j = 1; j < C; j++) if (row[j] > m) m = row[j];

        /* Steps 2-3: exp values + sum */
        int32_t S = 0;
        for (int j = 0; j < C; j++) {
            int diff = (int)row[j] - (int)m;
            if (diff < -255) diff = -255;
            int idx = diff + 255;
            uint8_t e = exp_lut[idx];
            orow[j] = e;            /* temp: store e in output */
            S += (int32_t)e;
        }

        /* Step 5: normalise */
        int32_t half_S = S / 2;
        for (int j = 0; j < C; j++) {
            int32_t ej = (int32_t)orow[j];
            orow[j] = (uint8_t)((ej * 255 + half_S) / S);
        }
    }
}
