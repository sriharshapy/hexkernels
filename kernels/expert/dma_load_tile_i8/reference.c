/* Baseline: scalar identity copy (no HVX, no VTCM staging).
 * This is the speedup denominator. Correct, but pays full DDR latency and
 * moves one byte at a time. */
#include <stdint.h>
void candidate_kernel(const int8_t *a, int8_t *out, int n) {
    for (int i = 0; i < n; i++) out[i] = a[i];
}
