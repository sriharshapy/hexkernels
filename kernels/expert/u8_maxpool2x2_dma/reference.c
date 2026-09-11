/* Baseline: naive scalar 2x2 max-pool. Speedup denominator: correct but no
 * vectorization, no VTCM staging -- pays full DDR latency and scalar
 * throughput on this large image. */
#include <stdint.h>
void candidate_kernel(const uint8_t *in, uint8_t *out, int w, int h) {
    int ow = w / 2, oh = h / 2;
    for (int oy = 0; oy < oh; oy++) {
        const uint8_t *r0 = in + (2*oy)   * w;
        const uint8_t *r1 = in + (2*oy+1) * w;
        uint8_t *o = out + oy * ow;
        for (int ox = 0; ox < ow; ox++) {
            int a = r0[2*ox], b = r0[2*ox+1], c = r1[2*ox], d = r1[2*ox+1];
            int m = a > b ? a : b; int n = c > d ? c : d; o[ox] = (uint8_t)(m > n ? m : n);
        }
    }
}
