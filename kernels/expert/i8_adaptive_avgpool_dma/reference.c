/* Baseline: scalar adaptive average pool (no HVX, no VTCM staging).
 * Per window, an exact int32 sum, then truncated mean cast to int8.
 * Speedup denominator. */
#include <stdint.h>
void candidate_kernel(const uint8_t *in, int8_t *out, int n, int m) {
    int k = n / m;
    for (int i = 0; i < m; i++) {
        const uint8_t *w = in + (long)i * k;
        int32_t s = 0;
        for (int j = 0; j < k; j++) s += (int32_t)w[j];
        out[i] = (int8_t)(s / k);
    }
}
