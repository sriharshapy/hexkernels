/* Near-miss: skips the 1/sqrt(d) scaling step before softmax.
 * Uses raw dot product values (int32 clamped to int8) directly.
 * For non-trivial inv_sqrt_d this produces wrong softmax distributions. */
#include <stdint.h>
#define SEQ_Q    16
#define SEQ_K    16
#define HEAD_DIM 32
void candidate_kernel(const int8_t *Q, const int8_t *K,
                      uint8_t *out,
                      const uint8_t *exp_lut,
                      int32_t inv_sqrt_d, int shift) {
    (void)inv_sqrt_d; (void)shift;  /* BUG: scaling ignored */
    int8_t scores[SEQ_Q*SEQ_K];
    for (int i = 0; i < SEQ_Q; i++) {
        for (int j = 0; j < SEQ_K; j++) {
            int32_t raw = 0;
            for (int d = 0; d < HEAD_DIM; d++)
                raw += (int32_t)Q[i*HEAD_DIM+d] * (int32_t)K[j*HEAD_DIM+d];
            /* BUG: no scaling -- clamp raw directly */
            if (raw >  127) raw =  127;
            if (raw < -128) raw = -128;
            scores[i*SEQ_K+j] = (int8_t)raw;
        }
    }
    for (int i = 0; i < SEQ_Q; i++) {
        const int8_t *row = scores + i * SEQ_K;
        uint8_t *orow = out + i * SEQ_K;
        int8_t m = row[0];
        for (int j = 1; j < SEQ_K; j++) if (row[j] > m) m = row[j];
        int32_t S = 0;
        for (int j = 0; j < SEQ_K; j++) {
            int diff = (int)row[j] - (int)m;
            if (diff < -255) diff = -255;
            uint8_t e = exp_lut[diff + 255];
            orow[j] = e;
            S += (int32_t)e;
        }
        int32_t half_S = S / 2;
        for (int j = 0; j < SEQ_K; j++) {
            int32_t ej = (int32_t)orow[j];
            orow[j] = (uint8_t)((ej * 255 + half_S) / S);
        }
    }
}
