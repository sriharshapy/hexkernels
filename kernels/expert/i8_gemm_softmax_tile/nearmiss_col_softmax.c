/* Near-miss: applies softmax over COLUMNS instead of rows.
 * Each column is normalized independently instead of each row. */
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
            if (acc >  127) acc =  127;
            if (acc < -128) acc = -128;
            scores[i*GN+j] = (int8_t)acc;
        }
    }
    /* BUG: column softmax instead of row softmax */
    for (int j = 0; j < GN; j++) {
        int8_t m = scores[0*GN+j];
        for (int i = 1; i < GM; i++) if (scores[i*GN+j] > m) m = scores[i*GN+j];
        int32_t S = 0;
        for (int i = 0; i < GM; i++) {
            int diff = (int)scores[i*GN+j] - (int)m;
            if (diff < -255) diff = -255;
            uint8_t e = exp_lut[diff + 255];
            out[i*GN+j] = e;
            S += (int32_t)e;
        }
        int32_t half_S = S / 2;
        for (int i = 0; i < GM; i++) {
            int32_t ej = (int32_t)out[i*GN+j];
            out[i*GN+j] = (uint8_t)((ej * 255 + half_S) / S);
        }
    }
}
