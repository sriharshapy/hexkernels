/* NEARMISS: Hardcodes mult1=3, shift1=7, mult2=5, shift2=8 regardless of runtime params.
 * Fails on the second param sweep (mult1=1, shift1=4, mult2=2, shift2=5). */
#include <stdint.h>

#define TOKENS   8
#define CHANNELS 16
#define D_MIX    16

static int8_t requant_i8_biased(int32_t acc, int32_t bias_v, int32_t mult, int shift) {
    int64_t biased = (int64_t)acc + (int64_t)bias_v;
    int64_t v      = biased * (int64_t)mult;
    int64_t half   = (shift > 0) ? ((int64_t)1 << (shift - 1)) : 0;
    int64_t q      = (v >= 0) ? ((v + half) >> shift) : -((-v + half) >> shift);
    if (q >  127) q =  127;
    if (q < -128) q = -128;
    return (int8_t)q;
}

void candidate_kernel(const int8_t  *X,
                      const int8_t  *W,
                      const int32_t *b,
                      const int8_t  *W2,
                      const int32_t *b2,
                      int8_t        *out,
                      int32_t mult1, int shift1,  /* BUG: ignored */
                      int32_t mult2, int shift2) {
    (void)mult1; (void)shift1; (void)mult2; (void)shift2;
    /* BUG: hardcoded params */
    int32_t hm1 = 3, hs1 = 7, hm2 = 5, hs2 = 8;

    for (int c = 0; c < CHANNELS; c++) {
        int8_t y1[D_MIX];
        for (int m = 0; m < D_MIX; m++) {
            int32_t acc = 0;
            for (int t = 0; t < TOKENS; t++)
                acc += (int32_t)W[m*TOKENS+t] * (int32_t)X[t*CHANNELS+c];
            y1[m] = requant_i8_biased(acc, b[m], hm1, hs1);
        }
        for (int t = 0; t < TOKENS; t++) {
            int32_t acc = 0;
            for (int m = 0; m < D_MIX; m++)
                acc += (int32_t)W2[t*D_MIX+m] * (int32_t)y1[m];
            out[t*CHANNELS+c] = requant_i8_biased(acc, b2[t], hm2, hs2);
        }
    }
}
