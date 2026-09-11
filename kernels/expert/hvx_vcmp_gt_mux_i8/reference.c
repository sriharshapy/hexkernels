#include <stdint.h>
/* Scalar baseline: compare a>b, select a or c. */
void candidate_kernel(const int8_t *a, const int8_t *b, const int8_t *c, int8_t *out, int n) {
    for (int i = 0; i < n; i++)
        out[i] = (a[i] > b[i]) ? a[i] : c[i];
}
