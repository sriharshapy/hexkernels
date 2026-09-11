/* Baseline: scalar saturating add-constant over DDR (no HVX, no VTCM
 * staging). This is the speedup denominator. */
#include <stdint.h>
void candidate_kernel(const uint8_t *a, uint8_t *out, int n) {
    for (int i = 0; i < n; i++) { int v = (int)a[i] + 50; out[i] = (uint8_t)(v > 255 ? 255 : v); }
}
