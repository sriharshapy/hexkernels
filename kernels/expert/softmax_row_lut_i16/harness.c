#include "harness_common.h"
#include "kernel_api.h"
#include <math.h>
#include <stdint.h>

#define R 6
#define C 113   /* NOT a multiple of 128 */
#define TOTAL (R*C)

static int16_t  x[TOTAL]     HVX_ALIGN;
static int16_t  out[TOTAL]   HVX_ALIGN;
static int16_t  ref[TOTAL]   HVX_ALIGN;
static uint16_t exp_lut[256] HVX_ALIGN;

/* Reference: same pinned formula, independent of candidate/baseline. */
static void softmax_row_ref(const int16_t *xv, int16_t *r, int rows, int cols,
                            const uint16_t *lut) {
    for (int row = 0; row < rows; row++) {
        const int16_t *xr  = xv + (long)row * cols;
        int16_t       *orv = r  + (long)row * cols;
        int16_t m = xr[0];
        for (int j = 1; j < cols; j++) if (xr[j] > m) m = xr[j];
        int32_t S = 0;
        for (int j = 0; j < cols; j++) {
            int32_t diff = (int32_t)xr[j] - (int32_t)m;
            if (diff < -255) diff = -255;
            uint16_t e = lut[diff + 255];
            S += (int32_t)e;
        }
        int32_t half_S = S / 2;
        for (int j = 0; j < cols; j++) {
            int32_t diff = (int32_t)xr[j] - (int32_t)m;
            if (diff < -255) diff = -255;
            uint16_t e = lut[diff + 255];
            int64_t num = (int64_t)e * 32767 + half_S;
            orv[j] = (int16_t)(num / S);
        }
    }
}

#define NSETS 3

int main(void) {
    uint32_t s = 0x9E1B2C3Du;

    /* Real exp-decay table (genuine dynamic range, runtime-built, 3 distinct
     * scales -> anti-hardcode). exp_lut[i] represents exp((i-255)/scale),
     * rescaled into uint16. */
    static const double scales[NSETS] = {35.0, 45.0, 60.0};
    uint16_t luts[NSETS][256];
    for (int k = 0; k < NSETS; k++) {
        for (int i = 0; i < 256; i++) {
            double val = 65535.0 * exp(((double)i - 255.0) / scales[k]);
            if (val < 1.0) val = 1.0;
            if (val > 65535.0) val = 65535.0;
            luts[k][i] = (uint16_t)(val + 0.5);
        }
        luts[k][255] = 65535; /* exact top entry (diff=0 -> e^0=1 scaled) */
    }

    /* Random input across the FULL int16 range. */
    for (int i = 0; i < TOTAL; i++) x[i] = (int16_t)(hvx_lcg(&s) & 0xFFFFu);

    /* Edge rows injected. */
    /* Row 0: all identical -> diff=0 everywhere, idx=255 (uniform softmax). */
    for (int j = 0; j < C; j++) x[0*C + j] = 5000;
    /* Row 1: one dominant max, rest at INT16_MIN -> diff = -32768-32767,
     * far beyond -255 (exercises the clamp path with a huge underflow). */
    for (int j = 0; j < C; j++) x[1*C + j] = -32768;
    x[1*C + 0] = 32767;
    /* Row 2: alternating extremes -> clamp path on (about) every other elem. */
    for (int j = 0; j < C; j++) x[2*C + j] = (j & 1) ? (int16_t)-32768 : (int16_t)32767;
    /* Row 3: mild negative ramp (no clamp triggered) -- dense non-clamped path. */
    for (int j = 0; j < C; j++) x[3*C + j] = (int16_t)(100 - j);

    int errors = 0, fb = -1;
    long gotv = 0, expv = 0;

    for (int k = 0; k < NSETS; k++) {
        for (int j = 0; j < 256; j++) exp_lut[j] = luts[k][j];

        softmax_row_ref(x, ref, R, C, exp_lut);

        for (int i = 0; i < TOTAL; i++) out[i] = (int16_t)0xA5A5;

        unsigned long long _hvx_kc = 0;
        HVX_TIME_KERNEL(_hvx_kc, { candidate_kernel(x, out, R, C, exp_lut); });
        printf("HVXENV_KCYCLES kernel=%llu\n", _hvx_kc);

        for (int i = 0; i < TOTAL; i++) {
            if (out[i] != ref[i]) {
                errors++;
                if (fb < 0) {
                    fb = k * TOTAL + i;
                    gotv = (long)out[i];
                    expv = (long)ref[i];
                }
            }
        }
    }

    hvx_report(errors, TOTAL * NSETS, fb, gotv, expv);
    return errors ? 1 : 0;
}
