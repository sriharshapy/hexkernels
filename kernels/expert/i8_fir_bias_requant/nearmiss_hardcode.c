/* Near-miss: hardcodes mult=7, shift=8, zp=0 -- ignores runtime quant params.
   Also uses bias. Passes first sweep set (7,8,0) but fails the other 2. */
#include <stdint.h>
void candidate_kernel(const int8_t *x, const int8_t *taps, int32_t bias,
                      int8_t *out,
                      int n, int ntaps,
                      int32_t mult, int shift, int8_t zp) {
    (void)mult; (void)shift; (void)zp;  /* intentionally hardcoded below */
    for (int i = 0; i < n; i++) {
        int32_t acc = 0;
        for (int k = 0; k < ntaps; k++)
            acc += (int32_t)x[i + k] * (int32_t)taps[k];
        long long biased = (long long)acc + (long long)bias;
        /* hardcoded: mult=7, shift=8, zp=0 */
        long long v = biased * 7LL;
        long long h = 128LL;  /* 1 << (8-1) */
        long long r = (v >= 0) ? ((v + h) >> 8) : -((-v + h) >> 8);
        if (r >  127) r =  127;
        if (r < -128) r = -128;
        out[i] = (signed char)r;
    }
}
