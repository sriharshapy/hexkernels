#include <stdint.h>
/* Scalar baseline: signed int8 elementwise max. */
void candidate_kernel(const int8_t *a, const int8_t *b, int8_t *out, int n) {
    for (int i = 0; i < n; i++)
        out[i] = (a[i] > b[i]) ? a[i] : b[i];
}
