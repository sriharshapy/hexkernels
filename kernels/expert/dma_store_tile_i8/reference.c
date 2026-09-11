/* Baseline: scalar bitwise-NOT reading/writing DDR directly (no HVX, no
 * VTCM staging). This is the speedup denominator. */
#include <stdint.h>
void candidate_kernel(const int8_t *a, int8_t *out, int n) {
    for (int i = 0; i < n; i++) out[i] = (int8_t)(~a[i]);
}
