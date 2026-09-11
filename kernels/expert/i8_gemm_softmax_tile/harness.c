#include "harness_common.h"
#include "kernel_api.h"
#include <stdint.h>
#include <string.h>

static int8_t  A[GM*GK]      HVX_ALIGN;
static int8_t  B[GK*GN]      HVX_ALIGN;
static uint8_t out[GM*GN]    HVX_ALIGN;
static uint8_t ref[GM*GN]    HVX_ALIGN;
static uint8_t exp_lut[256]  HVX_ALIGN;

/* Reference: GEMM -> clamp int8 -> rowwise softmax */
static void gemm_softmax_ref(const int8_t *a, const int8_t *b,
                             uint8_t *r, const uint8_t *lut) {
    /* Step A: GEMM */
    int8_t scores[GM*GN];
    for (int i = 0; i < GM; i++) {
        for (int j = 0; j < GN; j++) {
            int32_t acc = 0;
            for (int k = 0; k < GK; k++)
                acc += (int32_t)a[i*GK+k] * (int32_t)b[k*GN+j];
            /* Step B: clamp to int8 */
            if (acc >  127) acc =  127;
            if (acc < -128) acc = -128;
            scores[i*GN+j] = (int8_t)acc;
        }
    }

    /* Step C: rowwise softmax */
    for (int i = 0; i < GM; i++) {
        const int8_t *row = scores + i * GN;
        uint8_t      *or_ = r      + i * GN;
        int8_t m = row[0];
        for (int j = 1; j < GN; j++) if (row[j] > m) m = row[j];
        int32_t S = 0;
        for (int j = 0; j < GN; j++) {
            int diff = (int)row[j] - (int)m;
            if (diff < -255) diff = -255;
            uint8_t e = lut[diff + 255];
            or_[j] = e;
            S += (int32_t)e;
        }
        int32_t half_S = S / 2;
        for (int j = 0; j < GN; j++) {
            int32_t ej = (int32_t)or_[j];
            or_[j] = (uint8_t)((ej * 255 + half_S) / S);
        }
    }
}

/*
 * Anti-cheat: 3 distinct LUT tables.
 * A and B remain constant across sweeps -- only LUT changes.
 */
#define NSETS 3

int main(void) {
    uint32_t s = 0xD1B3A902u;

    uint8_t luts[NSETS][256];
    for (int k = 0; k < NSETS; k++) {
        for (int j = 0; j < 256; j++)
            luts[k][j] = (uint8_t)((hvx_lcg(&s) >> 24) & 0xFF);
        luts[k][255] = 200 + (uint8_t)(k * 25);
        luts[k][0]   = 1;
    }

    /* Random A and B */
    for (int i = 0; i < GM*GK; i++) A[i] = (int8_t)(hvx_lcg(&s) >> 24);
    for (int i = 0; i < GK*GN; i++) B[i] = (int8_t)(hvx_lcg(&s) >> 24);

    /* Edge cases */
    /* Row 0 of A all-max, col 0 of B all-max -> large positive acc */
    for (int k = 0; k < GK; k++) { A[0*GK+k] = 127; B[k*GN+0] = 127; }
    /* Row 1 of A all-min, col 1 of B all-max -> large negative acc */
    for (int k = 0; k < GK; k++) { A[1*GK+k] = -128; B[k*GN+1] = 127; }
    /* Row 2 of A all-zero -> all acc=0 for that row */
    for (int k = 0; k < GK; k++) A[2*GK+k] = 0;

    int errors = 0, fb = -1;
    long gotv = 0, expv = 0;

    for (int k = 0; k < NSETS; k++) {
        for (int j = 0; j < 256; j++) exp_lut[j] = luts[k][j];

        gemm_softmax_ref(A, B, ref, exp_lut);

        for (int i = 0; i < GM*GN; i++) out[i] = 0xA5;

        unsigned long long _hvx_kc = 0;
        HVX_TIME_KERNEL(_hvx_kc, { candidate_kernel(A, B, out, exp_lut); });
        printf("HVXENV_KCYCLES kernel=%llu\n", _hvx_kc);

        for (int i = 0; i < GM*GN; i++) {
            if (out[i] != ref[i]) {
                errors++;
                if (fb < 0) {
                    fb = k * GM * GN + i;
                    gotv = (long)out[i];
                    expv = (long)ref[i];
                }
            }
        }
    }

    hvx_report(errors, GM * GN * NSETS, fb, gotv, expv);
    return errors ? 1 : 0;
}
