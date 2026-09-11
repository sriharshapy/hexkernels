/* Baseline: pure scalar overlapping energy-pool (window=8, stride=4) +
 * shift/clamp. For each output j: acc = sum of squares of the 8-sample
 * window starting at x[j*4]; out[j] = min(127, acc >> shift) (acc >= 0 so
 * no lower clamp is needed). Speedup denominator. */
#include <stdint.h>
#include <stddef.h>

void candidate_kernel(const int8_t *x, int8_t *out, int n, int window, int shift) {
    (void)window;
    for (int j = 0; j < n; j++) {
        int32_t a = 0;
        for (int k = 0; k < 8; k++) { int32_t v = x[(size_t)j * 4 + k]; a += v * v; }
        int32_t r = a >> shift;
        if (r > 127) r = 127;
        out[j] = (int8_t)r;
    }
}
