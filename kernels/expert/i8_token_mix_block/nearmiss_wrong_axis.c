/* NEARMISS: Transposes the wrong axis -- mixes channels instead of tokens.
 * Instead of W[D_MIX x TOKENS] * X[:,c] (per column c), computes
 * W[D_MIX x TOKENS] * X[t,:] (per row t) -- wrong: this mixes channels not tokens. */
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
                      int32_t mult1, int shift1,
                      int32_t mult2, int shift2) {
    /* BUG: Iterates over tokens (rows) instead of channels (columns) in step1.
     * W has shape [D_MIX x TOKENS] but we index it as [D_MIX x CHANNELS] -- wrong dims.
     * This mixes on the wrong axis and produces incorrect outputs. */
    for (int t = 0; t < TOKENS; t++) {
        /* Step 1: incorrectly mixes channels: sum_c W[m,c]*X[t,c] instead of sum_t W[m,t]*X[t,c] */
        int8_t y1[D_MIX];
        for (int m = 0; m < D_MIX; m++) {
            int32_t acc = 0;
            /* BUG: indexing W by channel dimension (CHANNELS) not token dimension (TOKENS) */
            for (int c = 0; c < TOKENS; c++)  /* WRONG: should iterate over all TOKENS per channel */
                acc += (int32_t)W[m*TOKENS+c] * (int32_t)X[t*CHANNELS+c];
            y1[m] = requant_i8_biased(acc, b[m], mult1, shift1);
        }

        /* Step 2: project -- but y1 is wrong so output is wrong */
        for (int c = 0; c < CHANNELS; c++) {
            int32_t acc = 0;
            for (int m = 0; m < D_MIX; m++)
                acc += (int32_t)W2[t*D_MIX+m] * (int32_t)y1[m];
            out[t*CHANNELS+c] = requant_i8_biased(acc, b2[t], mult2, shift2);
        }
    }
}
