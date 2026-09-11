#include <stdint.h>
/* Scalar baseline: bitwise AND, byte-pattern (sign-agnostic). */
void candidate_kernel(const int8_t *a, const int8_t *b, int8_t *out, int n) {
    for (int i = 0; i < n; i++)
        out[i] = (int8_t)((uint8_t)a[i] & (uint8_t)b[i]);
}
