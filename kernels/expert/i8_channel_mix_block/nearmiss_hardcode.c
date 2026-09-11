/* NEARMISS: Hardcodes gelu_lut as identity (gelu_lut[i] = i-128) instead of reading
 * the runtime table. Will not match the harness's randomly-generated LUT tables. */
#include <stdint.h>

#define TOKENS   8
#define CHANNELS 16
#define D_FF     16

static int8_t requant_i8(int32_t acc, int32_t bias_v, int32_t mult, int shift) {
    int64_t biased = (int64_t)acc + (int64_t)bias_v;
    int64_t v      = biased * (int64_t)mult;
    int64_t half   = (shift > 0) ? ((int64_t)1 << (shift - 1)) : 0;
    int64_t q      = (v >= 0) ? ((v + half) >> shift) : -((-v + half) >> shift);
    if (q >  127) q =  127;
    if (q < -128) q = -128;
    return (int8_t)q;
}

void candidate_kernel(const int8_t  *X,
                      const int8_t  *W1,
                      const int32_t *b1,
                      const int8_t  *W2,
                      const int32_t *b2,
                      const int8_t  *gelu_lut,  /* BUG: ignored */
                      int8_t        *out,
                      int32_t mult1, int shift1,
                      int32_t mult2, int shift2) {
    (void)gelu_lut;
    for (int t = 0; t < TOKENS; t++) {
        int8_t y1[D_FF];
        for (int m = 0; m < D_FF; m++) {
            int32_t acc = 0;
            for (int c = 0; c < CHANNELS; c++)
                acc += (int32_t)W1[m*CHANNELS+c] * (int32_t)X[t*CHANNELS+c];
            int8_t sat1 = requant_i8(acc, b1[m], mult1, shift1);
            /* BUG: hardcoded identity LUT: idx -> idx-128 (i.e., LUT[idx]=idx-128) */
            uint8_t idx = (uint8_t)((int)sat1 + 128);
            y1[m] = (int8_t)((int)idx - 128);  /* wrong for any non-identity table */
        }

        for (int c = 0; c < CHANNELS; c++) {
            int32_t acc = 0;
            for (int m = 0; m < D_FF; m++)
                acc += (int32_t)W2[c*D_FF+m] * (int32_t)y1[m];
            out[t*CHANNELS+c] = requant_i8(acc, b2[c], mult2, shift2);
        }
    }
}
