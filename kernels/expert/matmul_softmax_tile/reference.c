#include <stdint.h>
#define SEQ_Q    16
#define SEQ_K    16
#define HEAD_DIM 32
/*
 * Matmul-softmax tile baseline -- pure integer, no FP.
 *
 * Step A: Q*K^T -> scale by (inv_sqrt_d >> shift) -> clamp to int8 scores.
 * Step B: rowwise softmax over int8 scores using exp_lut -> uint8 output.
 */
void candidate_kernel(const int8_t *Q, const int8_t *K,
                      uint8_t *out,
                      const uint8_t *exp_lut,
                      int32_t inv_sqrt_d, int shift) {
    /* Step A: compute scaled QK^T scores (int8 clamped) */
    int8_t scores[SEQ_Q*SEQ_K];
    for (int i = 0; i < SEQ_Q; i++) {
        for (int j = 0; j < SEQ_K; j++) {
            int32_t raw = 0;
            for (int d = 0; d < HEAD_DIM; d++)
                raw += (int32_t)Q[i*HEAD_DIM+d] * (int32_t)K[j*HEAD_DIM+d];
            /* round-half-up fixed-point scale */
            long long v    = (long long)raw * (long long)inv_sqrt_d;
            long long half = (shift > 0) ? ((long long)1 << (shift - 1)) : 0;
            long long sc;
            if (v >= 0) sc = (v + half) >> shift;
            else        sc = -((-v + half) >> shift);
            /* clamp to int8 */
            if (sc >  127) sc =  127;
            if (sc < -128) sc = -128;
            scores[i*SEQ_K+j] = (int8_t)sc;
        }
    }

    /* Step B: rowwise softmax */
    for (int i = 0; i < SEQ_Q; i++) {
        const int8_t *row = scores + i * SEQ_K;
        uint8_t      *orow = out   + i * SEQ_K;

        /* find row max */
        int8_t m = row[0];
        for (int j = 1; j < SEQ_K; j++) if (row[j] > m) m = row[j];

        /* exp + sum */
        int32_t S = 0;
        for (int j = 0; j < SEQ_K; j++) {
            int diff = (int)row[j] - (int)m;
            if (diff < -255) diff = -255;
            uint8_t e = exp_lut[diff + 255];
            orow[j] = e;
            S += (int32_t)e;
        }

        /* normalize */
        int32_t half_S = S / 2;
        for (int j = 0; j < SEQ_K; j++) {
            int32_t ej = (int32_t)orow[j];
            orow[j] = (uint8_t)((ej * 255 + half_S) / S);
        }
    }
}
