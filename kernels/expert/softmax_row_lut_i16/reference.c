#include <stdint.h>
/*
 * Row-wise softmax baseline (int16 in/out) -- pure integer, no FP.
 * For each row r in [0,R):
 *   1. m_r    = max over j in [0,C)
 *   2. idx_j  = clamp((int32_t)(x[r*C+j] - m_r), -255, 0) + 255
 *   3. e_j    = exp_lut[idx_j]                                  (uint16)
 *   4. S_r    = sum of e_j                                       (int32)
 *   5. out[r*C+j] = (int16_t)(((int64_t)e_j*32767 + S_r/2) / S_r)
 */
void candidate_kernel(const int16_t *x, int16_t *out, int R, int C,
                      const uint16_t *exp_lut) {
    for (int r = 0; r < R; r++) {
        const int16_t *row  = x   + (long)r * C;
        int16_t       *orow = out + (long)r * C;

        /* Step 1: row max */
        int16_t m = row[0];
        for (int j = 1; j < C; j++) if (row[j] > m) m = row[j];

        /* Steps 2-4: idx + exp lut + sum */
        int32_t S = 0;
        for (int j = 0; j < C; j++) {
            int32_t diff = (int32_t)row[j] - (int32_t)m;
            if (diff < -255) diff = -255;
            int idx = (int)diff + 255;
            uint16_t e = exp_lut[idx];
            S += (int32_t)e;
        }

        /* Step 5: normalise (recompute idx/e; avoids a temp buffer) */
        int32_t half_S = S / 2;
        for (int j = 0; j < C; j++) {
            int32_t diff = (int32_t)row[j] - (int32_t)m;
            if (diff < -255) diff = -255;
            int idx = (int)diff + 255;
            uint16_t e = exp_lut[idx];
            int64_t num = (int64_t)e * 32767 + half_S;
            orow[j] = (int16_t)(num / S);
        }
    }
}
