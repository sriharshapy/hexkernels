/* Near-miss: hardcodes mult=5, shift=3 for all rows, ignoring runtime arrays.
   Fails because harness sweeps multiple mult/shift array sets (anti-hardcode gate). */
#include <stdint.h>
void candidate_kernel(const int32_t *a, int8_t *out, int R, int C,
                      const int32_t *mult, const int *shift, int8_t zp) {
    (void)mult; (void)shift;
    for (int i = 0; i < R * C; i++) {
        long long v = (long long)a[i] * 5;
        long long h = 4;
        long long r = (v >= 0) ? ((v + h) >> 3) : -(((-v) + h) >> 3);
        r += zp;
        if (r > 127) r = 127;
        if (r < -128) r = -128;
        out[i] = (signed char)r;
    }
}
