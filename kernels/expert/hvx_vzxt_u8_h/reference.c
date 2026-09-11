#include <stdint.h>
/* Scalar baseline: per-block even/odd deinterleaved zero-extend. */
void candidate_kernel(const uint8_t *a, uint16_t *out, int n) {
    for (int base = 0; base + 128 <= n; base += 128) {
        for (int j = 0; j < 64; j++) {
            out[base + j]      = (uint16_t)a[base + 2*j];
            out[base + 64 + j] = (uint16_t)a[base + 2*j + 1];
        }
    }
}
