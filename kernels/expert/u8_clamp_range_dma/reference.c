/* Baseline: naive scalar clamp to [64,191]. Speedup denominator: correct but
 * no vectorization, no VTCM staging -- pays full DDR latency and scalar
 * throughput on every element. */
#include <stdint.h>
#define LO 64
#define HI 191
void candidate_kernel(const uint8_t *in, uint8_t *out, int n) {
    for (int i = 0; i < n; i++) {
        uint8_t x = in[i];
        if (x < LO) x = LO; else if (x > HI) x = HI;
        out[i] = x;
    }
}
