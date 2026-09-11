/* Near-miss: uses an int32 row-sum accumulator instead of int64 -- a very
 * plausible bug given the reference task this is adapted from (small-C
 * softmax_row_lut_i16) used int32 safely. At this task's C=65500, a
 * near-uniform row drives S_r up to ~65500*65535 (~4.29e9), which OVERFLOWS
 * signed int32 and produces visibly wrong normalized output for such rows.
 *
 * S is declared `volatile`: without it, -O2 can (legally, since signed
 * integer overflow is UB in C) algebraically "self-heal" downstream uses of
 * S via the no-overflow assumption, silently producing the mathematically
 * correct answer despite the wrapped bit pattern -- verified empirically on
 * this toolchain (a plain non-volatile int32_t accumulator here compiled to
 * code that happened to still pass). `volatile` forces the compiler to
 * treat every read of S as the actual (wrapped) stored value, so the bug
 * reliably manifests. */
#include <stdint.h>
#include <stddef.h>
#include <hexagon_types.h>
#include <hexagon_protos.h>

#define MAX_C 65500
static uint16_t escratch[MAX_C];

static void softmax_row_bad(const int16_t *xr, int16_t *outr, int C,
                            const uint16_t *lut) {
    int16_t m = xr[0];
    for (int j = 1; j < C; j++) if (xr[j] > m) m = xr[j];

    volatile int32_t S = 0;   /* BUG: should be int64 -- overflows at this C */
    for (int j = 0; j < C; j++) {
        int32_t diff = (int32_t)xr[j] - (int32_t)m;
        if (diff < -255) diff = -255;
        uint16_t e = lut[diff + 255];
        escratch[j] = e;
        S += (int32_t)e;
    }

    int32_t halfS = S / 2;
    for (int j = 0; j < C; j++) {
        int64_t num = (int64_t)escratch[j] * 32767 + halfS;
        outr[j] = (int16_t)(num / S);
    }
}

void candidate_kernel(const int16_t *x, int16_t *out, int R, int C,
                      const uint16_t *exp_lut) {
    for (int r = 0; r < R; r++)
        softmax_row_bad(x + (size_t)r * C, out + (size_t)r * C, C, exp_lut);
}
