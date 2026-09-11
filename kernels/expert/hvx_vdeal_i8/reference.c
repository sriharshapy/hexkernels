#include <stdint.h>
/* Scalar baseline: per-block even/odd de-interleave. */
void candidate_kernel(const int8_t *a, int8_t *out, int n) {
    for (int base = 0; base + 128 <= n; base += 128) {
        for (int i = 0; i < 64; i++) {
            out[base + i]      = a[base + 2*i];
            out[base + 64 + i] = a[base + 2*i + 1];
        }
    }
}
