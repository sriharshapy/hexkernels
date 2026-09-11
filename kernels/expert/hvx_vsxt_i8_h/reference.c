#include <stdint.h>
/* Scalar baseline: per-block even/odd deinterleaved sign-extend. */
void candidate_kernel(const int8_t *a, int16_t *out, int n) {
    for (int base = 0; base + 128 <= n; base += 128) {
        for (int j = 0; j < 64; j++) {
            out[base + j]      = (int16_t)a[base + 2*j];
            out[base + 64 + j] = (int16_t)a[base + 2*j + 1];
        }
    }
}
