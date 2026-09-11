#include <stdint.h>
/* Scalar baseline: two's-complement int16 wraparound subtract. */
void candidate_kernel(const int16_t *a, const int16_t *b, int16_t *out, int n) {
    for (int i = 0; i < n; i++)
        out[i] = (int16_t)((int)a[i] - (int)b[i]);
}
