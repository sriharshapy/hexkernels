/* Baseline: scalar int8->int16 dequantize over DDR (no VTCM staging, no HVX).
 * (x-zp)*scale >> shift, arithmetic right shift toward -inf, saturate to int16.
 * Speedup denominator. */
#include <stdint.h>

static int16_t ref_scalar(int8_t x, int8_t zp, int32_t scale, int shift) {
    int32_t v = ((int32_t)x - (int32_t)zp) * scale;
    int32_t r = v >> shift;
    if (r > 32767)  r = 32767;
    if (r < -32768) r = -32768;
    return (int16_t)r;
}

void candidate_kernel(const int8_t *a, int16_t *out, int n,
                      int8_t zp, int32_t scale, int shift) {
    for (int i = 0; i < n; i++) out[i] = ref_scalar(a[i], zp, scale, shift);
}
