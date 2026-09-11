/* Baseline: scalar saturating add against the cyclically-broadcast operand,
 * reading a[]/writing out[] directly against DDR (no HVX, no VTCM staging).
 * This is the speedup denominator. */
#include <stdint.h>

void candidate_kernel(const int8_t *a, const int8_t *op, int8_t *out, int n) {
    for (int i = 0; i < n; i++) {
        int32_t t = (int32_t)a[i] + (int32_t)op[i % 128];
        if (t > 127) t = 127; if (t < -128) t = -128;
        out[i] = (int8_t)t;
    }
}
