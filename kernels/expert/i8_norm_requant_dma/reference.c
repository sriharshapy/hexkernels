/* Baseline: pure scalar per-channel normalize + requantize (int8 -> int8).
 * Per-channel arrays and requant params baked (NUM_CH=16, MULT=5, SHIFT=4, ZP=0);
 * round half away from zero at each shift; sat8 on the final clamp. Denominator. */
#include <stdint.h>

#define NUM_CH 16
#define MULT  5
#define SHIFT 4
#define ZP    0

static const int32_t NORM_MULT[NUM_CH]  = { 3, 5, 7, 9, 11, 13, 15, 17, 3, 5, 7, 9, 11, 13, 15, 17 };
static const int     NORM_SHIFT[NUM_CH] = { 2, 3, 4, 5,  3,  4,  5,  6, 2, 3, 4, 5,  3,  4,  5,  6 };

static int8_t ref_scalar(int8_t ai, int32_t nm, int ns) {
    int64_t nv    = (int64_t)ai * (int64_t)nm;
    int64_t nhalf = ns > 0 ? ((int64_t)1 << (ns - 1)) : 0;
    int64_t norm  = (nv >= 0) ? ((nv + nhalf) >> ns) : -(((-nv) + nhalf) >> ns);
    int64_t v     = norm * (int64_t)MULT;
    int64_t half  = SHIFT > 0 ? ((int64_t)1 << (SHIFT - 1)) : 0;
    int64_t r     = (v >= 0) ? ((v + half) >> SHIFT) : -(((-v) + half) >> SHIFT);
    r += ZP;
    if (r >  127) r =  127;
    if (r < -128) r = -128;
    return (int8_t)r;
}

void candidate_kernel(const int8_t *a, int8_t *out, int n) {
    int per_ch = n / NUM_CH;
    for (int c = 0; c < NUM_CH; c++) {
        int32_t nm = NORM_MULT[c]; int ns = NORM_SHIFT[c];
        const int8_t *row = a + c * per_ch;
        int8_t *dst = out + c * per_ch;
        for (int i = 0; i < per_ch; i++) dst[i] = ref_scalar(row[i], nm, ns);
    }
}
