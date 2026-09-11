/* Near-miss: uses a single shared mult/shift instead of per-channel arrays.
   Fails because different rows have different mult[r]/shift[r]. */
#include <stdint.h>
void candidate_kernel(const int32_t *a, int8_t *out, int R, int C,
                      const int32_t *mult, const int *shift, int8_t zp) {
    /* WRONG: reads only mult[0]/shift[0] for all rows */
    int32_t m = mult[0];
    int     s = shift[0];
    int64_t half = s > 0 ? ((int64_t)1 << (s - 1)) : 0;
    for (int i = 0; i < R * C; i++) {
        int64_t v = (int64_t)a[i] * (int64_t)m;
        int64_t r = (v >= 0) ? ((v + half) >> s) : -(((-v) + half) >> s);
        r += zp;
        if (r > 127) r = 127;
        if (r < -128) r = -128;
        out[i] = (int8_t)r;
    }
}
