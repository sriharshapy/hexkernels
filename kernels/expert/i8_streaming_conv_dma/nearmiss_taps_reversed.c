/* Near-miss: convolution (taps REVERSED) instead of correlation. Uses
 * taps[ntaps-1-j] so the result differs whenever the taps are not
 * palindromic (they are not, in this task's harness). Scalar-only; compiles
 * and runs but must fail bit-exact. */
#include <stdint.h>
void candidate_kernel(const int8_t *x, const int8_t *taps, int8_t *out,
                      int n, int ntaps, int shift) {
    for (int i = 0; i < n; i++) {
        int32_t acc = 0;
        for (int j = 0; j < ntaps; j++) acc += (int32_t)x[i + j] * (int32_t)taps[ntaps - 1 - j];
        int32_t half = shift > 0 ? (1 << (shift - 1)) : 0;
        int32_t r = (acc >= 0) ? ((acc + half) >> shift) : -(((-acc) + half) >> shift);
        if (r > 127) r = 127; if (r < -128) r = -128;
        out[i] = (int8_t)r;
    }
}
