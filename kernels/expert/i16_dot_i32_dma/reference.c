/* Scalar int16 dot product over DDR (no VTCM staging, no vectorization). A
 * plain scalar loop accumulating into int64 (matching the harness's int64
 * reference accumulator) then truncated to int32 on store. Speedup
 * denominator. */
#include <stdint.h>

void candidate_kernel(const int16_t *a, const int16_t *b, int n, int32_t *out) {
    int64_t s = 0;
    for (int i = 0; i < n; i++) s += (int64_t)a[i] * (int64_t)b[i];
    out[0] = (int32_t)s;
}
