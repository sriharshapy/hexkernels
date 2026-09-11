/* Near-miss: ignores the dilation parameter, always uses stride 1.
   Only correct when dilation==1; fails for dilation==2 and dilation==3. */
#include <stdint.h>
void candidate_kernel(const int8_t *x, const int8_t *taps, int8_t *out,
                      int n, int ntaps, int dilation,
                      int32_t mult, int shift, int8_t zp) {
    (void)dilation;  /* BUG: ignores dilation, always uses stride 1 */
    for (int i = 0; i < n; i++) {
        int32_t acc = 0;
        for (int k = 0; k < ntaps; k++)
            acc += (int32_t)x[i + k] * (int32_t)taps[k];  /* stride 1 regardless */
        int64_t v    = (int64_t)acc * (int64_t)mult;
        int64_t half = shift > 0 ? ((int64_t)1 << (shift - 1)) : 0;
        int64_t r    = (v >= 0) ? ((v + half) >> shift) : -(((-v) + half) >> shift);
        r += zp;
        if (r >  127) r =  127;
        if (r < -128) r = -128;
        out[i] = (int8_t)r;
    }
}
