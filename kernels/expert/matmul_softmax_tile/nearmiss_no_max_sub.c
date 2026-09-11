/* Near-miss: does the QK^T scaling correctly but skips max-subtraction in softmax.
 * Uses raw score byte values as LUT indices -- numerically unstable and incorrect. */
#include <stdint.h>
#define SEQ_Q    16
#define SEQ_K    16
#define HEAD_DIM 32
void candidate_kernel(const int8_t *Q, const int8_t *K,
                      uint8_t *out,
                      const uint8_t *exp_lut,
                      int32_t inv_sqrt_d, int shift) {
    int8_t scores[SEQ_Q*SEQ_K];
    for (int i = 0; i < SEQ_Q; i++) {
        for (int j = 0; j < SEQ_K; j++) {
            int32_t raw = 0;
            for (int d = 0; d < HEAD_DIM; d++)
                raw += (int32_t)Q[i*HEAD_DIM+d] * (int32_t)K[j*HEAD_DIM+d];
            long long v    = (long long)raw * (long long)inv_sqrt_d;
            long long half = (shift > 0) ? ((long long)1 << (shift - 1)) : 0;
            long long sc;
            if (v >= 0) sc = (v + half) >> shift;
            else        sc = -((-v + half) >> shift);
            if (sc >  127) sc =  127;
            if (sc < -128) sc = -128;
            scores[i*SEQ_K+j] = (int8_t)sc;
        }
    }
    /* BUG: no max-subtract; uses (uint8_t)score as LUT index */
    for (int i = 0; i < SEQ_Q; i++) {
        const int8_t *row = scores + i * SEQ_K;
        uint8_t *orow = out + i * SEQ_K;
        int32_t S = 0;
        for (int j = 0; j < SEQ_K; j++) {
            uint8_t e = exp_lut[(uint8_t)row[j]];
            orow[j] = e;
            S += (int32_t)e;
        }
        if (S == 0) S = 1;
        int32_t half_S = S / 2;
        for (int j = 0; j < SEQ_K; j++) {
            int32_t ej = (int32_t)orow[j];
            orow[j] = (uint8_t)((ej * 255 + half_S) / S);
        }
    }
}
