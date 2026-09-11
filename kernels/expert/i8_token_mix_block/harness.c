#include "harness_common.h"
#include "kernel_api.h"
#include <stdint.h>
#include <string.h>

/* TOKENS=8, CHANNELS=16, D_MIX=16 */

static int8_t  X[TOKENS*CHANNELS]   HVX_ALIGN;
static int8_t  W[D_MIX*TOKENS]      HVX_ALIGN;
static int32_t b[D_MIX]             HVX_ALIGN;
static int8_t  W2[TOKENS*D_MIX]     HVX_ALIGN;
static int32_t b2[TOKENS]           HVX_ALIGN;
static int8_t  out[TOKENS*CHANNELS] HVX_ALIGN;
static int8_t  ref[TOKENS*CHANNELS] HVX_ALIGN;

static int8_t requant_i8_biased(int32_t acc, int32_t bias_v, int32_t mult, int shift) {
    int64_t biased = (int64_t)acc + (int64_t)bias_v;
    int64_t v      = biased * (int64_t)mult;
    int64_t half   = (shift > 0) ? ((int64_t)1 << (shift - 1)) : 0;
    int64_t q      = (v >= 0) ? ((v + half) >> shift) : -((-v + half) >> shift);
    if (q >  127) q =  127;
    if (q < -128) q = -128;
    return (int8_t)q;
}

static void token_mix_ref(const int8_t *Xv, const int8_t *Wv, const int32_t *bv,
                          const int8_t *W2v, const int32_t *b2v, int8_t *r,
                          int32_t mult1, int shift1, int32_t mult2, int shift2) {
    /* For each channel independently */
    for (int c = 0; c < CHANNELS; c++) {
        /* Step 1: token-mix GEMM W[D_MIX x TOKENS] * X[:,c] -> y1[D_MIX] */
        int8_t y1[D_MIX];
        for (int m = 0; m < D_MIX; m++) {
            int32_t acc = 0;
            for (int t = 0; t < TOKENS; t++)
                acc += (int32_t)Wv[m*TOKENS+t] * (int32_t)Xv[t*CHANNELS+c];
            y1[m] = requant_i8_biased(acc, bv[m], mult1, shift1);
        }

        /* Step 2: project back W2[TOKENS x D_MIX] * y1 -> out[:,c] */
        for (int t = 0; t < TOKENS; t++) {
            int32_t acc = 0;
            for (int m = 0; m < D_MIX; m++)
                acc += (int32_t)W2v[t*D_MIX+m] * (int32_t)y1[m];
            r[t*CHANNELS+c] = requant_i8_biased(acc, b2v[t], mult2, shift2);
        }
    }
}

/*
 * Sweep 2 distinct (mult1, shift1, mult2, shift2) param sets.
 * Bias arrays are data (not params) -- same across sweeps; params swept.
 */
static const int32_t MULT1S[]  = { 3, 1 };
static const int     SHIFT1S[] = { 7, 4 };
static const int32_t MULT2S[]  = { 5, 2 };
static const int     SHIFT2S[] = { 8, 5 };
#define NSETS 2

int main(void) {
    uint32_t s = 0xC1D2E3F4u;

    /* Random inputs */
    for (int i = 0; i < TOKENS*CHANNELS; i++) X[i]  = (int8_t)(hvx_lcg(&s) >> 24);
    for (int i = 0; i < D_MIX*TOKENS;   i++) W[i]  = (int8_t)(hvx_lcg(&s) >> 24);
    for (int i = 0; i < D_MIX;          i++) b[i]  = (int32_t)((hvx_lcg(&s) & 0xFFFF) - 0x8000);
    for (int i = 0; i < TOKENS*D_MIX;   i++) W2[i] = (int8_t)(hvx_lcg(&s) >> 24);
    for (int i = 0; i < TOKENS;         i++) b2[i] = (int32_t)((hvx_lcg(&s) & 0xFFFF) - 0x8000);

    /* Edge cases */
    X[0] = 127;  W[0] = 127;    /* large positive product */
    X[1] = -128; W[1] = 127;    /* sign edge in GEMM */
    b[0] = -200000;              /* large negative bias -> potential saturation */
    b[1] =  200000;              /* large positive bias */
    b2[0] = -200000;
    b2[1] =  200000;
    W2[0] = 127; W2[1] = -128;  /* sign edge in second GEMM */

    int errors = 0, fb = -1;
    long gotv = 0, expv = 0;

    for (int p = 0; p < NSETS; p++) {
        int32_t mult1  = MULT1S[p];
        int     shift1 = SHIFT1S[p];
        int32_t mult2  = MULT2S[p];
        int     shift2 = SHIFT2S[p];

        token_mix_ref(X, W, b, W2, b2, ref, mult1, shift1, mult2, shift2);

        for (int i = 0; i < TOKENS*CHANNELS; i++) out[i] = (int8_t)0xA5;

        unsigned long long _hvx_kc = 0;
        HVX_TIME_KERNEL(_hvx_kc, { candidate_kernel(X, W, b, W2, b2, out, mult1, shift1, mult2, shift2); });
        printf("HVXENV_KCYCLES kernel=%llu\n", _hvx_kc);

        for (int i = 0; i < TOKENS*CHANNELS; i++) {
            if (out[i] != ref[i]) {
                errors++;
                if (fb < 0) {
                    fb = p * TOKENS*CHANNELS + i;
                    gotv = (long)out[i];
                    expv = (long)ref[i];
                }
            }
        }
    }

    hvx_report(errors, NSETS * TOKENS*CHANNELS, fb, gotv, expv);
    return errors ? 1 : 0;
}
