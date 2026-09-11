#include <stdint.h>
void candidate_kernel(const int32_t *a, const int32_t *b, int8_t *out, int n,
                      int32_t mult, int shift, int8_t zp) {
    for (int i = 0; i < n; i++) {
        /* Step 1: add in 64-bit to avoid overflow */
        int64_t sum = (int64_t)a[i] + (int64_t)b[i];
        /* Step 2: relu */
        if (sum < 0) sum = 0;
        /* Step 3: requantize round-half-away-from-zero */
        int64_t v    = sum * (int64_t)mult;
        int64_t half = shift > 0 ? ((int64_t)1 << (shift - 1)) : 0;
        int64_t r    = (v >= 0) ? ((v + half) >> shift) : -(((-v) + half) >> shift);
        r += zp;
        if (r >  127) r =  127;
        if (r < -128) r = -128;
        out[i] = (int8_t)r;
    }
}
