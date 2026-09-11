#include <stdint.h>
#define RT 37
/* Scalar baseline: cross-block window at fixed offset RT, combined={b:a}. */
void candidate_kernel(const int8_t *a, const int8_t *b, int8_t *out, int n) {
    for (int base = 0; base + 128 <= n; base += 128) {
        for (int i = 0; i < 128; i++) {
            int idx = RT + i;
            out[base + i] = (idx < 128) ? b[base + idx] : a[base + idx - 128];
        }
    }
}
