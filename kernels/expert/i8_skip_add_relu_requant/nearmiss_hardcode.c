/* Near-miss: hardcodes mult=5, shift=4, zp=0 -- ignores runtime params.
   Passes only the first sweep set but FAILS the other four. */
#include <stdint.h>
void candidate_kernel(const int8_t *a, const int8_t *b, int8_t *out, int n,
                      int32_t mult, int shift, int8_t zp) {
    (void)mult; (void)shift; (void)zp;
    for (int i = 0; i < n; i++) {
        int32_t sum = (int32_t)a[i] + (int32_t)b[i];
        if (sum < 0) sum = 0;
        /* hardcoded mult=5, shift=4, zp=0 */
        int64_t v = (int64_t)sum * 5LL;
        int64_t r = (v + 8LL) >> 4;
        if (r > 127) r = 127;
        out[i] = (int8_t)r;
    }
}
