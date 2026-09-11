#include <stdint.h>
/*
 * Row-wise reciprocal-LUT softmax baseline -- pure integer, no FP, no
 * division. Per row r:
 *   1. m    = max over j
 *   2. idx_j = clamp(x[r*C+j]-m, -255, 0) + 255
 *   3. e_j  = exp_lut[idx_j]
 *   4. S    = sum e_j
 *   5. sidx = clamp(S>>6, 0, 255)
 *   6. recip = recip_lut[sidx]
 *   7. out[r*C+j] = clamp((e_j*recip+32768)>>16, 0, 255)
 */
void candidate_kernel(const int8_t *x, uint8_t *out, int R, int C,
                      const uint8_t *exp_lut, const uint16_t *recip_lut) {
    for (int r = 0; r < R; r++) {
        const int8_t *row = x + r * C;
        uint8_t      *orow = out + r * C;

        int8_t m = row[0];
        for (int j = 1; j < C; j++) if (row[j] > m) m = row[j];

        int32_t S = 0;
        for (int j = 0; j < C; j++) {
            int diff = (int)row[j] - (int)m;
            if (diff < -255) diff = -255;
            if (diff > 0) diff = 0;
            int idx = diff + 255;
            uint8_t e = exp_lut[idx];
            orow[j] = e;          /* temp: store e_j in output */
            S += (int32_t)e;
        }

        int32_t sidx = S >> 6;
        if (sidx < 0) sidx = 0;
        if (sidx > 255) sidx = 255;
        int32_t recip = (int32_t)recip_lut[sidx];

        for (int j = 0; j < C; j++) {
            int32_t ej = (int32_t)orow[j];
            int32_t v  = (ej * recip + 32768) >> 16;
            if (v < 0) v = 0;
            if (v > 255) v = 255;
            orow[j] = (uint8_t)v;
        }
    }
}
