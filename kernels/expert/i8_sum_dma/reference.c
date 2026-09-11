/* Scalar baseline: int8 sum-reduction over DDR (no VTCM staging), int32
 * accumulator. Pure scalar reference; speedup denominator. */
#include <stdint.h>
void candidate_kernel(const int8_t *a, int n, int32_t *out) {
    int32_t s = 0;
    for (int i = 0; i < n; i++) s += (int32_t)a[i];
    out[0] = s;
}
