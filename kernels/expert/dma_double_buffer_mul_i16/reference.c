/* Baseline: scalar low-16 truncating multiply reading/writing DDR directly
 * (no HVX, no VTCM staging). This is the speedup denominator. */
#include <stdint.h>
void candidate_kernel(const int16_t *a, const int16_t *b, int16_t *out, int n) {
    for (int i = 0; i < n; i++) out[i] = (int16_t)((int32_t)a[i] * (int32_t)b[i]);
}
