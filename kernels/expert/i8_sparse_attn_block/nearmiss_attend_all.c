/* NEARMISS: Attends to ALL positions (ignores the window/mask constraint).
 * With the planted high-magnitude keys outside the window, this produces
 * different attention weights and different output. */
#include <stdint.h>

#define H        1
#define SEQ      8
#define HEAD_DIM 16

static int8_t requant_i8(int32_t raw, int32_t mult, int shift) {
    int64_t v    = (int64_t)raw * (int64_t)mult;
    int64_t half = (shift > 0) ? ((int64_t)1 << (shift - 1)) : 0;
    int64_t q    = (v >= 0) ? ((v + half) >> shift) : -((-v + half) >> shift);
    if (q >  127) q =  127;
    if (q < -128) q = -128;
    return (int8_t)q;
}

void candidate_kernel(const int8_t  *Q,
                      const int8_t  *K,
                      const int8_t  *V,
                      const uint8_t *exp_lut,
                      int8_t        *out,
                      int window,       /* BUG: ignored */
                      int32_t smult, int sshift,
                      int32_t amult, int ashift) {
    (void)window;  /* BUG: attends to all positions */
    for (int h = 0; h < H; h++) {
        const int8_t *Qh  = Q + h * SEQ * HEAD_DIM;
        const int8_t *Kh  = K + h * SEQ * HEAD_DIM;
        const int8_t *Vh  = V + h * SEQ * HEAD_DIM;
        int8_t       *outh = out + h * SEQ * HEAD_DIM;

        for (int i = 0; i < SEQ; i++) {
            /* No masking -- attend everywhere */
            int8_t scores[SEQ];
            for (int j = 0; j < SEQ; j++) {
                int32_t acc = 0;
                for (int d = 0; d < HEAD_DIM; d++)
                    acc += (int32_t)Qh[i*HEAD_DIM+d] * (int32_t)Kh[j*HEAD_DIM+d];
                scores[j] = requant_i8(acc, smult, sshift);
            }

            int8_t m = scores[0];
            for (int j = 1; j < SEQ; j++) if (scores[j] > m) m = scores[j];

            uint8_t e[SEQ];
            int32_t S = 0;
            for (int j = 0; j < SEQ; j++) {
                int diff = (int)scores[j] - (int)m;
                if (diff < -255) diff = -255;
                e[j] = exp_lut[diff + 255];
                S += (int32_t)e[j];
            }

            int32_t half_S = S / 2;
            uint8_t prob[SEQ];
            for (int j = 0; j < SEQ; j++)
                prob[j] = (uint8_t)(((int32_t)e[j] * 255 + half_S) / S);

            for (int dv = 0; dv < HEAD_DIM; dv++) {
                int32_t acc = 0;
                for (int j = 0; j < SEQ; j++)
                    acc += (int32_t)prob[j] * (int32_t)Vh[j*HEAD_DIM+dv];
                outh[i*HEAD_DIM+dv] = requant_i8(acc, amult, ashift);
            }
        }
    }
}
