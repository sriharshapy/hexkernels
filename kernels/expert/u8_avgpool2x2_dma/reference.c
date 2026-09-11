/* Baseline: naive scalar 2x2 truncated-average pool. Speedup denominator:
 * correct but no vectorization, no VTCM staging -- pays full DDR latency and
 * scalar throughput on this large image. */
#include <stdint.h>
void candidate_kernel(const uint8_t *in, uint8_t *out, int w, int h) {
    int ow = w / 2, oh = h / 2;
    for (int oy = 0; oy < oh; oy++) {
        const uint8_t *r0 = in + (2*oy)   * w;
        const uint8_t *r1 = in + (2*oy+1) * w;
        uint8_t *o = out + oy * ow;
        for (int ox = 0; ox < ow; ox++) {
            unsigned a=r0[2*ox], b=r0[2*ox+1], c=r1[2*ox], d=r1[2*ox+1];
            o[ox] = (uint8_t)((a+b+c+d)/4);
        }
    }
}
