#include <stdint.h>
/* Scalar baseline: two's-complement int8 wraparound add. */
void candidate_kernel(const int8_t *a, const int8_t *b, int8_t *out, int n) {
    for (int i = 0; i < n; i++)
        out[i] = (int8_t)((int)a[i] + (int)b[i]);
}
