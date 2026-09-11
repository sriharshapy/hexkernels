/* Near-miss: plain windowed SUM (not sum-of-SQUARES) before the shift/clamp,
 * using the correct window=8/stride=4 indexing. Plausible ("pooling") but
 * wrong reduction -- differs whenever a window has mixed-sign or nonzero
 * values (i.e. almost everywhere). Also omits the lower clamp a signed sum
 * would need, which is itself part of why this is wrong for the specified
 * (always-non-negative) energy-pool contract. */
#include <stdint.h>
void candidate_kernel(const int8_t *x, int8_t *out, int n, int window, int shift) {
    (void)window;
    for (int j = 0; j < n; j++) {
        int32_t acc = 0;
        for (int k = 0; k < 8; k++) acc += (int32_t)x[j * 4 + k];
        int32_t r = acc >> shift;
        if (r > 127) r = 127;
        out[j] = (int8_t)r;
    }
}
