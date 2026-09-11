/* Baseline: naive scalar inclusive threshold (thresh=128). Speedup
 * denominator: correct but no vectorization, no VTCM staging -- pays full DDR
 * latency and scalar throughput on every element. */
#include <stdint.h>
#define THRESH 128
void candidate_kernel(const uint8_t *in, uint8_t *out, int n) {
    for (int i = 0; i < n; i++) out[i] = (uint8_t)(in[i] >= THRESH ? 255 : 0);
}
