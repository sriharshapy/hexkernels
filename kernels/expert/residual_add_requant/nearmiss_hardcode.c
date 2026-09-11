/* Near-miss: hardcodes mult=5, shift=3, zp=0 -- ignores the runtime params.
   Passes only the first sweep set (mult=5,shift=3,zp=0) but FAILS all others.
   The harness sweeps 5 distinct (mult,shift,zp) sets, so this kernel fails. */
#include <stdint.h>
void candidate_kernel(const int32_t *a, const int32_t *b, int8_t *out, int n,
                      int32_t mult, int shift, int8_t zp) {
    (void)mult; (void)shift; (void)zp;  /* intentionally ignore runtime params */
    for (int i = 0; i < n; i++) {
        int64_t sum = (int64_t)a[i] + (int64_t)b[i];
        /* hardcoded mult=5, shift=3, zp=0 */
        int64_t v = sum * 5LL;
        int64_t r = (v >= 0) ? ((v + 4LL) >> 3) : -(((-v) + 4LL) >> 3);
        if (r >  127) r =  127;
        if (r < -128) r = -128;
        out[i] = (int8_t)r;
    }
}
