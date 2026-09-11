#include <stdint.h>
/* Scalar baseline: bitwise OR. */
void candidate_kernel(const uint8_t *a, const uint8_t *b, uint8_t *out, int n) {
    for (int i = 0; i < n; i++)
        out[i] = (uint8_t)(a[i] | b[i]);
}
