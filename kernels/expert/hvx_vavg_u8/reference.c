#include <stdint.h>
/* Scalar baseline: rounding (round-half-up) average of two uint8 values. */
void candidate_kernel(const uint8_t *a, const uint8_t *b, uint8_t *out, int n) {
    for (int i = 0; i < n; i++)
        out[i] = (uint8_t)(((int)a[i] + (int)b[i] + 1) >> 1);
}
