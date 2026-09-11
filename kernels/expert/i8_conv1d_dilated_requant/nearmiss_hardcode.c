/* Near-miss: hardcodes mult=4, shift=5, zp=0 -- ignores runtime quant params.
   Uses correct dilation. Passes first quant set (4,5,0) for all dilations,
   but fails the second set (1,0,7). */
#include <stdint.h>
void candidate_kernel(const int8_t *x, const int8_t *taps, int8_t *out,
                      int n, int ntaps, int dilation,
                      int32_t mult, int shift, int8_t zp) {
    (void)mult; (void)shift; (void)zp;  /* intentionally hardcoded below */
    for (int i = 0; i < n; i++) {
        int32_t acc = 0;
        for (int k = 0; k < ntaps; k++)
            acc += (int32_t)x[i + k * dilation] * (int32_t)taps[k];
        /* hardcoded: mult=4, shift=5, zp=0 */
        long long v = (long long)acc * 4LL;
        long long h = 16LL;  /* 1 << (5-1) */
        long long r = (v >= 0) ? ((v + h) >> 5) : -((-v + h) >> 5);
        if (r >  127) r =  127;
        if (r < -128) r = -128;
        out[i] = (signed char)r;
    }
}
