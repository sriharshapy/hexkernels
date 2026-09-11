/* Scalar baseline: int8 elementwise add, two's-complement wraparound.
 * out[i] = (int8_t)(a[i] + b[i]). */
#include <stdint.h>
void candidate_kernel(const int8_t *a, const int8_t *b, int8_t *out, int n) {
    for (int i = 0; i < n; i++)
        out[i] = (int8_t)(a[i] + b[i]);
}
