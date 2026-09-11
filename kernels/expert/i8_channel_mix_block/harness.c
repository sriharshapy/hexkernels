#include "harness_common.h"
#include "kernel_api.h"
#include <stdint.h>
#include <string.h>

/* TOKENS=8, CHANNELS=16, D_FF=16 */

static int8_t  X[TOKENS*CHANNELS]   HVX_ALIGN;
static int8_t  W1[D_FF*CHANNELS]    HVX_ALIGN;
static int32_t b1[D_FF]             HVX_ALIGN;
static int8_t  W2[CHANNELS*D_FF]    HVX_ALIGN;
static int32_t b2[CHANNELS]         HVX_ALIGN;
static int8_t  gelu_lut[256]        HVX_ALIGN;
static int8_t  out[TOKENS*CHANNELS] HVX_ALIGN;
static int8_t  ref[TOKENS*CHANNELS] HVX_ALIGN;

static int8_t requant_i8(int32_t acc, int32_t bias_v, int32_t mult, int shift) {
    int64_t biased = (int64_t)acc + (int64_t)bias_v;
    int64_t v      = biased * (int64_t)mult;
    int64_t half   = (shift > 0) ? ((int64_t)1 << (shift - 1)) : 0;
    int64_t q      = (v >= 0) ? ((v + half) >> shift) : -((-v + half) >> shift);
    if (q >  127) q =  127;
    if (q < -128) q = -128;
    return (int8_t)q;
}

static void channel_mix_ref(const int8_t *Xv, const int8_t *W1v, const int32_t *b1v,
                            const int8_t *W2v, const int32_t *b2v,
                            const int8_t *glut, int8_t *r,
                            int32_t mult1, int shift1, int32_t mult2, int shift2) {
    for (int t = 0; t < TOKENS; t++) {
        /* Step 1: expand GEMM + bias + requant + GELU LUT */
        int8_t y1[D_FF];
        for (int m = 0; m < D_FF; m++) {
            int32_t acc = 0;
            for (int c = 0; c < CHANNELS; c++)
                acc += (int32_t)W1v[m*CHANNELS+c] * (int32_t)Xv[t*CHANNELS+c];
            int8_t sat1 = requant_i8(acc, b1v[m], mult1, shift1);
            uint8_t idx = (uint8_t)((int)sat1 + 128);
            y1[m] = glut[idx];
        }

        /* Step 2: project back GEMM + bias + requant */
        for (int c = 0; c < CHANNELS; c++) {
            int32_t acc = 0;
            for (int m = 0; m < D_FF; m++)
                acc += (int32_t)W2v[c*D_FF+m] * (int32_t)y1[m];
            r[t*CHANNELS+c] = requant_i8(acc, b2v[c], mult2, shift2);
        }
    }
}

/*
 * Sweep: 2 distinct gelu_lut tables x 2 param sets = 4 evals.
 * Anti-hardcode: candidate must read runtime lut and params.
 */
#define NLUTS   2
#define NPARAMS 2
#define NSETS   (NLUTS * NPARAMS)

static const int32_t MULT1S[]  = { 3, 1 };
static const int     SHIFT1S[] = { 7, 4 };
static const int32_t MULT2S[]  = { 5, 2 };
static const int     SHIFT2S[] = { 8, 5 };

int main(void) {
    uint32_t s = 0xD4C3B2A1u;

    /* Random inputs */
    for (int i = 0; i < TOKENS*CHANNELS; i++) X[i]  = (int8_t)(hvx_lcg(&s) >> 24);
    for (int i = 0; i < D_FF*CHANNELS;   i++) W1[i] = (int8_t)(hvx_lcg(&s) >> 24);
    for (int i = 0; i < D_FF;            i++) b1[i] = (int32_t)((hvx_lcg(&s) & 0xFFFF) - 0x8000);
    for (int i = 0; i < CHANNELS*D_FF;   i++) W2[i] = (int8_t)(hvx_lcg(&s) >> 24);
    for (int i = 0; i < CHANNELS;        i++) b2[i] = (int32_t)((hvx_lcg(&s) & 0xFFFF) - 0x8000);

    /* Edge cases */
    X[0] = 127;  W1[0] = 127;   /* large positive product */
    X[1] = -128; W1[1] = 127;   /* sign edge */
    b1[0] = -200000;             /* large negative bias */
    b1[1] =  200000;             /* large positive bias */
    b2[0] = -200000;
    b2[1] =  200000;

    /* Build 2 distinct gelu_lut tables (random but distinct) */
    int8_t luts[NLUTS][256];
    for (int li = 0; li < NLUTS; li++) {
        for (int j = 0; j < 256; j++)
            luts[li][j] = (int8_t)(hvx_lcg(&s) >> 24);
        luts[li][128] = 0;             /* GELU(0) = 0 in both */
        luts[li][255] = 127;           /* GELU(127) ~ 127 */
        luts[li][0]   = (int8_t)(-128 + li*20); /* distinct min per table */
    }

    int errors = 0, fb = -1;
    long gotv = 0, expv = 0;

    for (int li = 0; li < NLUTS; li++) {
        for (int j = 0; j < 256; j++) gelu_lut[j] = luts[li][j];

        for (int pi = 0; pi < NPARAMS; pi++) {
            int32_t mult1  = MULT1S[pi];
            int     shift1 = SHIFT1S[pi];
            int32_t mult2  = MULT2S[pi];
            int     shift2 = SHIFT2S[pi];

            channel_mix_ref(X, W1, b1, W2, b2, gelu_lut, ref,
                            mult1, shift1, mult2, shift2);

            for (int i = 0; i < TOKENS*CHANNELS; i++) out[i] = (int8_t)0xA5;

            unsigned long long _hvx_kc = 0;
            HVX_TIME_KERNEL(_hvx_kc, {
                candidate_kernel(X, W1, b1, W2, b2, gelu_lut, out,
                mult1, shift1, mult2, shift2);
            });
            printf("HVXENV_KCYCLES kernel=%llu\n", _hvx_kc);

            for (int i = 0; i < TOKENS*CHANNELS; i++) {
                if (out[i] != ref[i]) {
                    errors++;
                    if (fb < 0) {
                        fb = (li*NPARAMS + pi) * TOKENS*CHANNELS + i;
                        gotv = (long)out[i];
                        expv = (long)ref[i];
                    }
                }
            }
        }
    }

    hvx_report(errors, NSETS * TOKENS*CHANNELS, fb, gotv, expv);
    return errors ? 1 : 0;
}
