/* softmax_row_i8 harness. Row-wise softmax via a division-free quantized
 * reciprocal LUT (no integer division anywhere, unlike softmax_rowwise).
 * Harness owns main() and computes the scalar reference INDEPENDENTLY
 * (same pinned formula, duplicated here -- never calls baseline.c/expert.c). */
#include "harness_common.h"
#include "kernel_api.h"
#include <math.h>
#include <stdint.h>

#define R 8
#define C 100   /* NOT a multiple of 128 */
#define TOTAL (R*C)

static int8_t  x[TOTAL]     HVX_ALIGN;
static uint8_t out[TOTAL]   HVX_ALIGN;
static uint8_t ref[TOTAL]   HVX_ALIGN;
static uint8_t  exp_lut[256]   HVX_ALIGN;
static uint16_t recip_lut[256] HVX_ALIGN;

int main(void) {
    uint32_t s = 0x50F7A21u;

    /* exp_lut[idx] = clamp(round(255*exp((idx-255)/32.0)), 1, 255) -- generated
     * ONCE by a fixed formula (not randomized; anti-hardcode is satisfied
     * because the candidate only ever sees this via the runtime pointer). */
    for (int idx = 0; idx < 256; idx++) {
        double v = 255.0 * exp(((double)idx - 255.0) / 32.0);
        long iv = (long)(v + 0.5);
        if (iv < 1) iv = 1;
        if (iv > 255) iv = 255;
        exp_lut[idx] = (uint8_t)iv;
    }

    /* recip_lut[sidx] = clamp(round(65536.0/center), 1, 65535), center = max(1,(sidx<<6)+32) */
    for (int sidx = 0; sidx < 256; sidx++) {
        long center = ((long)sidx << 6) + 32;
        if (center < 1) center = 1;
        double v = 65536.0 / (double)center;
        long iv = (long)(v + 0.5);
        if (iv < 1) iv = 1;
        if (iv > 65535) iv = 65535;
        recip_lut[sidx] = (uint16_t)iv;
    }

    /* Random input */
    for (int i = 0; i < TOTAL; i++) x[i] = (int8_t)(hvx_lcg(&s) >> 24);

    /* Row 0: all-identical values -> every diff=0, idx=255 for all j. */
    for (int j = 0; j < C; j++) x[0*C + j] = 17;

    /* Row 1: one dominant max, rest at -128 (extreme spread). */
    for (int j = 0; j < C; j++) x[1*C + j] = -128;
    x[1*C + 0] = 127;

    /* Row 2: alternating +127/-128 (dense clamp exercise). */
    for (int j = 0; j < C; j++) x[2*C + j] = (j & 1) ? -128 : 127;

    /* Independent scalar reference (same pinned formula). */
    for (int r = 0; r < R; r++) {
        const int8_t *row = x + r * C;
        uint8_t      *rr  = ref + r * C;

        int8_t m = row[0];
        for (int j = 1; j < C; j++) if (row[j] > m) m = row[j];

        int32_t S = 0;
        uint8_t erow[C];
        for (int j = 0; j < C; j++) {
            int diff = (int)row[j] - (int)m;
            if (diff < -255) diff = -255;
            if (diff > 0) diff = 0;
            uint8_t e = exp_lut[diff + 255];
            erow[j] = e;
            S += (int32_t)e;
        }
        int32_t sidx = S >> 6;
        if (sidx < 0) sidx = 0;
        if (sidx > 255) sidx = 255;
        int32_t recip = (int32_t)recip_lut[sidx];

        for (int j = 0; j < C; j++) {
            int32_t ej = (int32_t)erow[j];
            int32_t v  = (ej * recip + 32768) >> 16;
            if (v < 0) v = 0;
            if (v > 255) v = 255;
            rr[j] = (uint8_t)v;
        }
    }

    for (int i = 0; i < TOTAL; i++) out[i] = 0xA5;

    unsigned long long _hvx_kc = 0;
    HVX_TIME_KERNEL(_hvx_kc, { candidate_kernel(x, out, R, C, exp_lut, recip_lut); });
    printf("HVXENV_KCYCLES kernel=%llu\n", _hvx_kc);

    int errors = 0, fb = -1;
    long gotv = 0, expv = 0;
    for (int i = 0; i < TOTAL; i++) {
        if (out[i] != ref[i]) {
            errors++;
            if (fb < 0) { fb = i; gotv = (long)out[i]; expv = (long)ref[i]; }
        }
    }

    hvx_report(errors, TOTAL, fb, gotv, expv);
    return errors ? 1 : 0;
}
