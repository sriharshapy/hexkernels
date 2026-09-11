/* Near-miss: skips the int8 clamp after GEMM; uses the raw int32 acc value
 * (cast to int8_t with C truncation, NOT saturating clamp).
 * C truncation (int8_t)(int32_t) truncates to low byte -- different from sat clamp. */
#include <stdint.h>
#define GM 16
#define GN 16
#define GK 16
void candidate_kernel(const int8_t *A, const int8_t *B,
                      uint8_t *out,
                      const uint8_t *exp_lut) {
    int8_t scores[GM*GN];
    for (int i = 0; i < GM; i++) {
        for (int j = 0; j < GN; j++) {
            int32_t acc = 0;
            for (int k = 0; k < GK; k++)
                acc += (int32_t)A[i*GK+k] * (int32_t)B[k*GN+j];
            /* BUG: truncation cast instead of saturating clamp */
            scores[i*GN+j] = (int8_t)acc;
        }
    }
    for (int i = 0; i < GM; i++) {
        const int8_t *row  = scores + i * GN;
        uint8_t      *orow = out    + i * GN;
        int8_t m = row[0];
        for (int j = 1; j < GN; j++) if (row[j] > m) m = row[j];
        int32_t S = 0;
        for (int j = 0; j < GN; j++) {
            int diff = (int)row[j] - (int)m;
            if (diff < -255) diff = -255;
            uint8_t e = exp_lut[diff + 255];
            orow[j] = e;
            S += (int32_t)e;
        }
        int32_t half_S = S / 2;
        for (int j = 0; j < GN; j++) {
            int32_t ej = (int32_t)orow[j];
            orow[j] = (uint8_t)((ej * 255 + half_S) / S);
        }
    }
}
