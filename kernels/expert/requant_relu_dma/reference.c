/* Scalar baseline: requantize int32 -> int8 with fused ReLU.
 * Params baked: MULT=13, SHIFT=3, ZP=0. Round half away from zero. */
#include <stdint.h>

#define MULT  13
#define SHIFT 3
#define ZP    0

static int8_t ref_scalar(int32_t x) {
    int64_t v = (int64_t)x * (int64_t)MULT;
    int64_t half = (SHIFT > 0) ? ((int64_t)1 << (SHIFT - 1)) : 0;
    int64_t r = (v >= 0) ? ((v + half) >> SHIFT) : -(((-v) + half) >> SHIFT);
    r += ZP;
    if (r < (int64_t)ZP) r = (int64_t)ZP;
    if (r > 127) r = 127;
    return (int8_t)r;
}

void candidate_kernel(const int32_t *a, int8_t *out, int n) {
    for (int i = 0; i < n; i++) out[i] = ref_scalar(a[i]);
}
