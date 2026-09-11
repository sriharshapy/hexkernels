#include <stdint.h>
/* Scalar baseline: per-block byte reversal. */
void candidate_kernel(const int8_t *a, int8_t *out, int n) {
    for (int base = 0; base + 128 <= n; base += 128) {
        for (int i = 0; i < 128; i++)
            out[base + i] = a[base + 127 - i];
    }
}
