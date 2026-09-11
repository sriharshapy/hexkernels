/* Baseline: the naive ~4-DDR-pass-per-row version (speedup denominator).
 * Pass 1: scan x (DDR) for the row max. Pass 2: re-read x (DDR), LUT-lookup
 * (scalar, inherently -- no vectorized gather), accumulate the int64 row sum,
 * and cache e_j into a plain STATIC scratch array (NOT VTCM, so still
 * DDR-cost). Pass 3: read the cached e_j (DDR-cost) and write out (DDR-cost).
 * No VTCM staging anywhere. */
#include <stdint.h>
#include <stddef.h>
#include <hexagon_types.h>
#include <hexagon_protos.h>

#define MAX_C 65500

static uint16_t escratch[MAX_C];   /* plain array -- DDR-cost, reused per row */

static void softmax_row_naive(const int16_t *xr, int16_t *outr, int C,
                              const uint16_t *lut) {
    /* Pass 1: max scan. */
    int16_t m = xr[0];
    for (int j = 1; j < C; j++) if (xr[j] > m) m = xr[j];

    /* Pass 2: re-read x, LUT lookup, accumulate int64 sum, cache e_j. */
    int64_t S = 0;
    for (int j = 0; j < C; j++) {
        int32_t diff = (int32_t)xr[j] - (int32_t)m;
        if (diff < -255) diff = -255;
        uint16_t e = lut[diff + 255];
        escratch[j] = e;
        S += (int64_t)e;
    }

    /* Pass 3: read cached e_j, normalize, write out. */
    int64_t halfS = S / 2;
    for (int j = 0; j < C; j++) {
        int64_t num = (int64_t)escratch[j] * 32767 + halfS;
        outr[j] = (int16_t)(num / S);
    }
}

void candidate_kernel(const int16_t *x, int16_t *out, int R, int C,
                      const uint16_t *exp_lut) {
    for (int r = 0; r < R; r++)
        softmax_row_naive(x + (size_t)r * C, out + (size_t)r * C, C, exp_lut);
}
