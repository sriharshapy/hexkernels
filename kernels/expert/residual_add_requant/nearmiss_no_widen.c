/* Near-miss: computes a[i]+b[i] in int32 without widening to int64.
   Overflows on large residual sums (e.g. a=INT32_MAX/2 + b=INT32_MAX/2 wraps).
   The harness injects a[3]=INT32_MAX/2, b[3]=INT32_MAX/2 -- this candidate
   gets a wrong result there due to 32-bit overflow. */
#include <stdint.h>
void candidate_kernel(const int32_t *a, const int32_t *b, int8_t *out, int n,
                      int32_t mult, int shift, int8_t zp) {
    for (int i = 0; i < n; i++) {
        /* BUG: no widening -- int32 sum can overflow */
        int32_t sum32 = a[i] + b[i];
        int64_t v    = (int64_t)sum32 * (int64_t)mult;
        int64_t half = shift > 0 ? ((int64_t)1 << (shift - 1)) : 0;
        int64_t r    = (v >= 0) ? ((v + half) >> shift) : -(((-v) + half) >> shift);
        r += zp;
        if (r >  127) r =  127;
        if (r < -128) r = -128;
        out[i] = (int8_t)r;
    }
}
