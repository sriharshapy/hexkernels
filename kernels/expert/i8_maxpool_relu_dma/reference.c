/* Baseline: pure scalar fused 2x2 signed max pool + ReLU reading/writing DDR
 * directly (no HVX, no VTCM staging). Correct but unvectorized. Speedup
 * denominator. */
#include <stdint.h>
#include "kernel_api.h"

void candidate_kernel(const int8_t *in, int8_t *out, int w, int h) {
    int ow = w / 2, oh = h / 2;
    for (int oy = 0; oy < oh; oy++) {
        const int8_t *r0 = in + (2*oy)   * w;
        const int8_t *r1 = in + (2*oy+1) * w;
        int8_t *o = out + oy * ow;
        for (int ox = 0; ox < ow; ox++) {
            int a = r0[2*ox], b = r0[2*ox+1], c = r1[2*ox], d = r1[2*ox+1];
            int m = a > b ? a : b, n = c > d ? c : d;
            int mm = m > n ? m : n;
            o[ox] = (int8_t)(mm > 0 ? mm : 0);
        }
    }
}
