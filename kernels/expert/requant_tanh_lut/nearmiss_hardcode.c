/* Near-miss: hardcodes mult=3, shift=3, zp=0 -- ignores runtime params.
   Passes only the first sweep set but FAILS the remaining 5 sweep+LUT combinations. */
#include <stdint.h>
void candidate_kernel(const int32_t *a, int8_t *out, int n,
                      int32_t mult, int shift, int8_t zp,
                      const int8_t lut[256]) {
    (void)mult; (void)shift; (void)zp;
    for (int i = 0; i < n; i++) {
        /* hardcoded mult=3, shift=3, zp=0 */
        int64_t v = (int64_t)a[i] * 3LL;
        int64_t r = (v >= 0) ? ((v + 4LL) >> 3) : -(((-v) + 4LL) >> 3);
        if (r < 0)   r = 0;
        if (r > 255) r = 255;
        out[i] = lut[(uint8_t)r];
    }
}
