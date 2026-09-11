#include "harness_common.h"
#include "kernel_api.h"
#include <stdint.h>

#define R 8
#define C 137   /* NOT a multiple of 128 */
#define TOTAL (R*C)

static int8_t  x[TOTAL]     HVX_ALIGN;
static uint8_t out[TOTAL]   HVX_ALIGN;
static uint8_t ref[TOTAL]   HVX_ALIGN;
static uint8_t exp_lut[256] HVX_ALIGN;

/* Reference: same pinned formula */
static void softmax_row_ref(const int8_t *xv, uint8_t *r, int rows, int cols,
                            const uint8_t *lut) {
    for (int row = 0; row < rows; row++) {
        const int8_t *xr = xv + row * cols;
        uint8_t      *or = r  + row * cols;
        int8_t m = xr[0];
        for (int j = 1; j < cols; j++) if (xr[j] > m) m = xr[j];
        int32_t S = 0;
        for (int j = 0; j < cols; j++) {
            int diff = (int)xr[j] - (int)m;
            if (diff < -255) diff = -255;
            uint8_t e = lut[diff + 255];
            or[j] = e;
            S += (int32_t)e;
        }
        int32_t half_S = S / 2;
        for (int j = 0; j < cols; j++) {
            int32_t ej = (int32_t)or[j];
            or[j] = (uint8_t)((ej * 255 + half_S) / S);
        }
    }
}

#define NSETS 3

int main(void) {
    uint32_t s = 0xABCD1234u;

    uint8_t luts[NSETS][256];
    for (int k = 0; k < NSETS; k++) {
        for (int j = 0; j < 256; j++)
            luts[k][j] = (uint8_t)((hvx_lcg(&s) >> 24) & 0xFF);
        luts[k][255] = 180 + (uint8_t)(k * 25);
        luts[k][0]   = 1;
    }

    /* Random input */
    for (int i = 0; i < TOTAL; i++) x[i] = (int8_t)(hvx_lcg(&s) >> 24);

    /* Edge rows injected */
    /* Row 0: all same value -> every diff=0, idx=255 */
    for (int j = 0; j < C; j++) x[0*C + j] = 42;
    /* Row 1: one max, rest min */
    for (int j = 0; j < C; j++) x[1*C + j] = -128;
    x[1*C + 0] = 127;
    /* Row 2: alternating +127/-128 */
    for (int j = 0; j < C; j++) x[2*C + j] = (j & 1) ? -128 : 127;

    int errors = 0, fb = -1;
    long gotv = 0, expv = 0;

    for (int k = 0; k < NSETS; k++) {
        for (int j = 0; j < 256; j++) exp_lut[j] = luts[k][j];

        softmax_row_ref(x, ref, R, C, exp_lut);

        for (int i = 0; i < TOTAL; i++) out[i] = 0xA5;

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
