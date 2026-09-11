/* Scalar baseline: rescale int16 -> int8. Params baked: MULT=3, SHIFT=4, ZP=0.
 * Round half away from zero. */
#include <stdint.h>

#define MULT  3
#define SHIFT 4
#define ZP    0

static int8_t ref_scalar(int16_t x) {
    int32_t v = (int32_t)x * MULT;
    int32_t half = (SHIFT > 0) ? (1 << (SHIFT - 1)) : 0;
    int32_t r = (v >= 0) ? ((v + half) >> SHIFT) : -(((-v) + half) >> SHIFT);
    r += (int32_t)ZP;
    if (r > 127)  r = 127;
    if (r < -128) r = -128;
    return (int8_t)r;
}

void candidate_kernel(const int16_t *a, int8_t *out, int n) {
    for (int i = 0; i < n; i++) out[i] = ref_scalar(a[i]);
}
