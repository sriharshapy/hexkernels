/* Near-miss: hardcodes mult=5, shift=6, zp=0 -- ignores runtime quant params.
   Passes first sweep set (5,6,0) but fails the other 2 sets. */
#include <stdint.h>
void candidate_kernel(const int8_t *x, const int8_t *taps, int8_t *out,
                      int C, int L, int K,
                      int32_t mult, int shift, int8_t zp) {
    (void)mult; (void)shift; (void)zp;  /* intentionally ignore runtime params */
    for (int c = 0; c < C; c++) {
        const int8_t *xc   = x    + c * (L + K - 1);
        const int8_t *tapc = taps + c * K;
        int8_t       *oc   = out  + c * L;
        for (int i = 0; i < L; i++) {
            int32_t acc = 0;
            for (int k = 0; k < K; k++)
                acc += (int32_t)xc[i + k] * (int32_t)tapc[k];
            /* hardcoded: mult=5, shift=6, zp=0 */
            long long v = (long long)acc * 5LL;
            long long h = 32LL;  /* 1 << (6-1) */
            long long r = (v >= 0) ? ((v + h) >> 6) : -((-v + h) >> 6);
            if (r >  127) r =  127;
            if (r < -128) r = -128;
            oc[i] = (signed char)r;
        }
    }
}
