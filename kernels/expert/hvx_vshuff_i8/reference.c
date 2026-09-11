#include <stdint.h>
/* Scalar baseline: per-block low/high-half interleave. */
void candidate_kernel(const int8_t *a, int8_t *out, int n) {
    for (int base = 0; base + 128 <= n; base += 128) {
        for (int i = 0; i < 64; i++) {
            out[base + 2*i]     = a[base + i];
            out[base + 2*i + 1] = a[base + 64 + i];
        }
    }
}
