/* vtcm_softmax_row_i16 harness (v6, Group C: hvx + dma + vtcm).
 * Row-wise int16 softmax via a runtime 256-entry LUT, adapted from
 * softmax_row_lut_i16 but with an int64 row-sum (S_r overflows int32 at this
 * C). Harness owns main(): maps VTCM identity, builds a single runtime
 * exp-decay LUT (real dynamic range, NOT hardcoded in the candidate), seeds
 * deterministic per-row-varying input plus the pinned edge-case rows
 * (uniform / one-dominant-max / alternating-extremes / bounded ramp),
 * computes the pinned-formula scalar reference directly, poisons out, times
 * the candidate (kernel-only pcycles), bit-exact compares.
 *
 * Size macros are named HR/HC (NOT R/C) so a fast-debug override like
 * -DHC=2000 on the compiler command line cannot collide with the
 * candidate's own R/C parameter identifiers (those files are compiled in
 * the same invocation as this harness).
 */
#include "harness_common.h"
#include "kernel_api.h"
#include "hexagon_standalone.h"
#include <hexagon_types.h>
#include <hexagon_protos.h>
#include <math.h>

#ifndef HR
#define HR 10
#endif
#ifndef HC
#define HC 65500   /* NOT a multiple of 128 -- tail path */
#endif
#define TOTAL ((size_t)HR * HC)

static int16_t  x[TOTAL]     HVX_ALIGN;
static int16_t  out[TOTAL]   HVX_ALIGN;
static int16_t  ref[TOTAL]   HVX_ALIGN;
static uint16_t exp_lut[256] HVX_ALIGN;

/* Reference: same pinned formula, int64 sum, independent of candidate/baseline. */
static void softmax_row_ref(const int16_t *xv, int16_t *r, int rows, int cols,
                            const uint16_t *lut) {
    for (int row = 0; row < rows; row++) {
        const int16_t *xr  = xv + (size_t)row * cols;
        int16_t       *orv = r  + (size_t)row * cols;
        int16_t m = xr[0];
        for (int j = 1; j < cols; j++) if (xr[j] > m) m = xr[j];
        int64_t S = 0;
        for (int j = 0; j < cols; j++) {
            int32_t diff = (int32_t)xr[j] - (int32_t)m;
            if (diff < -255) diff = -255;
            uint16_t e = lut[diff + 255];
            S += (int64_t)e;
        }
        int64_t half_S = S / 2;
        for (int j = 0; j < cols; j++) {
            int32_t diff = (int32_t)xr[j] - (int32_t)m;
            if (diff < -255) diff = -255;
            uint16_t e = lut[diff + 255];
            int64_t num = (int64_t)e * 32767 + half_S;
            orv[j] = (int16_t)(num / S);
        }
    }
}

int main(void) {
    uint32_t vtcm = __rdcfg(__vtcm_base) << 16;
    add_translation((void*)(uintptr_t)vtcm, (void*)(uintptr_t)vtcm, 0);

    uint32_t s = 0x9E1B2C3Du;

    /* Single runtime exp-decay table (genuine dynamic range, NOT hardcoded
     * in the candidate). exp_lut[i] represents exp((i-255)/scale), rescaled
     * into uint16. */
    const double scale = 45.0;
    for (int i = 0; i < 256; i++) {
        double val = 65535.0 * exp(((double)i - 255.0) / scale);
        if (val < 1.0) val = 1.0;
        if (val > 65535.0) val = 65535.0;
        exp_lut[i] = (uint16_t)(val + 0.5);
    }
    exp_lut[255] = 65535;   /* exact top entry (diff=0 -> e^0=1 scaled) */

    /* Random input across the FULL int16 range. */
    for (size_t i = 0; i < TOTAL; i++) x[i] = (int16_t)(hvx_lcg(&s) & 0xFFFFu);

    /* Edge rows (rows 0-3), overwriting the random fill for those rows. */
    /* Row 0: all identical -> diff=0 everywhere, idx=255 uniform softmax.
     * This is also the int32-overflow-inducing row for the nearmiss: S_r =
     * HC * 65535 ~= 4.29e9, which overflows int32. */
    for (int j = 0; j < HC; j++) x[(size_t)0 * HC + j] = 5000;
    /* Row 1: one dominant max, rest at INT16_MIN -> diff = -32768-32767,
     * far beyond -255 (huge underflow clamp path). */
    for (int j = 0; j < HC; j++) x[(size_t)1 * HC + j] = -32768;
    x[(size_t)1 * HC + 0] = 32767;
    /* Row 2: alternating extremes -> clamp path on (about) every other elem. */
    for (int j = 0; j < HC; j++)
        x[(size_t)2 * HC + j] = (j & 1) ? (int16_t)-32768 : (int16_t)32767;
    /* Row 3: bounded period-256 ramp (100+20 .. -135) -> diff stays within
     * [-255,0] at every element (dense NON-clamped path), unlike a naive
     * full-row linear ramp which would clamp almost everywhere at this C. */
    for (int j = 0; j < HC; j++) x[(size_t)3 * HC + j] = (int16_t)(120 - (j % 256));

    softmax_row_ref(x, ref, HR, HC, exp_lut);

    for (size_t i = 0; i < TOTAL; i++) out[i] = (int16_t)0xA5A5;   /* poison */

    unsigned long long _hvx_kc = 0;
    HVX_TIME_KERNEL(_hvx_kc, { candidate_kernel(x, out, HR, HC, exp_lut); });
    printf("HVXENV_KCYCLES kernel=%llu\n", _hvx_kc);

    int errors = 0; long fb = -1, gotv = 0, expv = 0;
    for (size_t i = 0; i < TOTAL; i++) {
        if (out[i] != ref[i]) {
            errors++;
            if (fb < 0) { fb = (long)i; gotv = (long)out[i]; expv = (long)ref[i]; }
        }
    }
    hvx_report(errors, (int)TOTAL, (int)fb, gotv, expv);
    return errors ? 1 : 0;
}
