/* Baseline: plain scalar fused 2x2 avg-pool (signed, trunc toward zero) +
 * requant, reading/writing DDR directly (no VTCM staging, no HVX). Speedup
 * denominator: correct but scalar, and reads the large image cold from DDR. */
#include <stdint.h>

static int8_t ref_element(int pool, int32_t mult, int shift, int8_t zp) {
    int64_t v    = (int64_t)pool * (int64_t)mult;
    int64_t half = shift > 0 ? ((int64_t)1 << (shift - 1)) : 0;
    int64_t r    = (v >= 0) ? ((v + half) >> shift) : -(((-v) + half) >> shift);
    r += zp;
    if (r >  127) r =  127;
    if (r < -128) r = -128;
    return (int8_t)r;
}

void candidate_kernel(const int8_t *in, int8_t *out, int w, int h,
                      int32_t mult, int shift, int8_t zp) {
    int ow = w / 2, oh = h / 2;
    for (int oy = 0; oy < oh; oy++) {
        const int8_t *r0 = in + (2*oy)   * w;
        const int8_t *r1 = in + (2*oy+1) * w;
        int8_t *o = out + oy * ow;
        for (int ox = 0; ox < ow; ox++) {
            int a = r0[2*ox], b = r0[2*ox+1], c = r1[2*ox], d = r1[2*ox+1];
            o[ox] = ref_element((a+b+c+d)/4, mult, shift, zp);
        }
    }
}
