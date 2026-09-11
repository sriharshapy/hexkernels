/* Near-miss: skips the ReLU step -- negative pre-activations flow through.
   Fails whenever a[i]+b[i] < 0 (happens frequently with the random inputs). */
#include <stdint.h>
void candidate_kernel(const int8_t *a, const int8_t *b, int8_t *out, int n,
                      int32_t mult, int shift, int8_t zp) {
    for (int i = 0; i < n; i++) {
        int32_t sum  = (int32_t)a[i] + (int32_t)b[i];
        /* BUG: no ReLU clamp -- negative sums propagate */
        int64_t v    = (int64_t)sum * (int64_t)mult;
        int64_t half = shift > 0 ? ((int64_t)1 << (shift - 1)) : 0;
        int64_t r    = (v >= 0) ? ((v + half) >> shift) : -(((-v) + half) >> shift);
        r += zp;
        if (r >  127) r =  127;
        if (r < -128) r = -128;
        out[i] = (int8_t)r;
    }
}
