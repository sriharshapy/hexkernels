/* Near-miss: correct FIR + requant but ignores the bias.
   Output differs from reference wherever bias shifts the value (bias=500 is non-trivial). */
#include <stdint.h>
void candidate_kernel(const int8_t *x, const int8_t *taps, int32_t bias,
                      int8_t *out,
                      int n, int ntaps,
                      int32_t mult, int shift, int8_t zp) {
    (void)bias;  /* BUG: intentionally ignored */
    for (int i = 0; i < n; i++) {
        int32_t acc = 0;
        for (int k = 0; k < ntaps; k++)
            acc += (int32_t)x[i + k] * (int32_t)taps[k];
        /* no bias */
        int64_t v    = (int64_t)acc * (int64_t)mult;
        int64_t half = shift > 0 ? ((int64_t)1 << (shift - 1)) : 0;
        int64_t r    = (v >= 0) ? ((v + half) >> shift) : -(((-v) + half) >> shift);
        r += zp;
        if (r >  127) r =  127;
        if (r < -128) r = -128;
        out[i] = (int8_t)r;
    }
}
