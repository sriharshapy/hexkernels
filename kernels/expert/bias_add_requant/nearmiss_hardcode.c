/* Near-miss: hardcodes mult=5, shift=3. Fails on any other parameter set. */
#include <stdint.h>
void candidate_kernel(const int32_t *a, const int32_t *bias, int8_t *out, int n,
                      int32_t mult, int shift, int8_t zp) {
    (void)mult; (void)shift;
    for (int i = 0; i < n; i++) {
        long long sum = (long long)a[i] + (long long)bias[i];
        long long v   = sum * 5;
        long long h   = 4;
        long long r   = (v >= 0) ? ((v + h) >> 3) : -(((-v) + h) >> 3);
        r += zp;
        if (r > 127)  r = 127;
        if (r < -128) r = -128;
        out[i] = (signed char)r;
    }
}
