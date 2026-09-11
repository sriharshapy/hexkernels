#include <stdint.h>
#define RT 19
/* Scalar baseline: cross-block window ending 128-RT into combined={b:a}. */
void candidate_kernel(const int8_t *a, const int8_t *b, int8_t *out, int n) {
    for (int base = 0; base + 128 <= n; base += 128) {
        for (int i = 0; i < 128; i++)
            out[base + i] = (i >= RT) ? a[base + i - RT] : b[base + 128 - RT + i];
    }
}
