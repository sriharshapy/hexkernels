/* Near-miss: left/right tap weights swapped (w[0] applied to the RIGHT
 * neighbor, w[2] applied to the LEFT neighbor) -- a classic tap-order bug.
 * Input data is effectively random (hence asymmetric), so with w[0]=2 !=
 * w[2]=3 this is wrong at almost every position. */
#include <stdint.h>

void candidate_kernel(const int8_t *a, const int8_t *w, int32_t bias, int shift,
                      int8_t *out, int n) {
    int32_t half = (shift > 0) ? (1 << (shift - 1)) : 0;
    for (int i = 0; i < n; i++) {
        int32_t left   = a[(i > 0) ? (i - 1) : 0];
        int32_t center = a[i];
        int32_t right  = a[(i < n - 1) ? (i + 1) : (n - 1)];
        /* BUG: w[0] applied to right, w[2] applied to left (swapped). */
        int32_t sum = (int32_t)w[2] * left + (int32_t)w[1] * center + (int32_t)w[0] * right + bias;
        int64_t absum = sum < 0 ? -(int64_t)sum : (int64_t)sum;
        int64_t sh = (absum + half) >> shift;
        int64_t r = sum < 0 ? -sh : sh;
        if (r > 127) r = 127; if (r < -128) r = -128;
        out[i] = (int8_t)r;
    }
}
