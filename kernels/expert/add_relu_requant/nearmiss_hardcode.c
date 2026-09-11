/* Near-miss: hardcodes mult=5,shift=4,zp=0 -- ignores the runtime params.
   Passes the first param-sweep iteration (mult=5,shift=4,zp=0) but FAILS others.
   The harness sweeps 5 distinct (mult,shift,zp) sets, so this kernel fails overall. */
#include <stdint.h>
void candidate_kernel(const int32_t *a, const int32_t *b, int8_t *out, int n,
                      int32_t mult, int shift, int8_t zp) {
    (void)mult; (void)shift; (void)zp;  /* intentionally ignore runtime params */
    for (int i = 0; i < n; i++) {
        int64_t sum = (int64_t)a[i] + (int64_t)b[i];
        if (sum < 0) sum = 0;
        /* hardcoded mult=5, shift=4, zp=0 */
        int64_t v = sum * 5LL;
        int64_t r = (v + 8LL) >> 4;  /* shift=4, half=8 */
        if (r >  127) r =  127;
        if (r < -128) r = -128;
        out[i] = (int8_t)r;
    }
}
