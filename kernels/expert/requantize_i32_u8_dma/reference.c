/* Scalar baseline: requantize int32 -> uint8 (asymmetric), round-half-away-
 * from-zero + zp, saturate to [0,255]. mult/shift/zp are runtime params. */
#include <stdint.h>

static uint8_t ref_scalar(int32_t ai, int32_t mult, int shift, uint8_t zp) {
    int64_t v = (int64_t)ai * (int64_t)mult;
    int64_t half = (shift > 0) ? ((int64_t)1 << (shift - 1)) : 0;
    int64_t r = (v >= 0) ? ((v + half) >> shift) : -(((-v) + half) >> shift);
    r += (int64_t)(uint64_t)zp;
    if (r > 255) r = 255;
    if (r < 0)   r = 0;
    return (uint8_t)r;
}

void candidate_kernel(const int32_t *a, uint8_t *out, int n,
                      int32_t mult, int shift, uint8_t zp) {
    for (int i = 0; i < n; i++) out[i] = ref_scalar(a[i], mult, shift, zp);
}
