/* Scalar baseline: fixed-point scale-by-1.5 with round-to-nearest, saturating
 * to int8: out[i] = clamp((a[i]*3 + 1) >> 1, -128, 127). */
#include <stdint.h>
void candidate_kernel(const int8_t *a, int8_t *out, int n) {
    for (int i = 0; i < n; i++) {
        int32_t t = (int32_t)a[i] * 3;
        t = (t + 1) >> 1;
        if (t > 127) t = 127;
        if (t < -128) t = -128;
        out[i] = (int8_t)t;
    }
}
