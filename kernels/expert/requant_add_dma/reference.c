/* Scalar baseline: residual-add requantize (int32+int32 -> int8).
 * Params baked: MULT=5, SHIFT=3, ZP=0. Round half away from zero. */
#include <stdint.h>

#define MULT  5
#define SHIFT 3
#define ZP    0

static int8_t ref_scalar(int32_t ax, int32_t bx) {
    int64_t sum = (int64_t)ax + (int64_t)bx;
    int64_t v   = sum * (int64_t)MULT;
    int64_t half = (SHIFT > 0) ? ((int64_t)1 << (SHIFT - 1)) : 0;
    int64_t r   = (v >= 0) ? ((v + half) >> SHIFT) : -(((-v) + half) >> SHIFT);
    r += ZP;
    if (r >  127) r =  127;
    if (r < -128) r = -128;
    return (int8_t)r;
}

void candidate_kernel(const int32_t *a, const int32_t *b, int8_t *out, int n) {
    for (int i = 0; i < n; i++) out[i] = ref_scalar(a[i], b[i]);
}
