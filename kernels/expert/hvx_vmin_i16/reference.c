#include <stdint.h>
/* Scalar baseline: signed int16 elementwise min. */
void candidate_kernel(const int16_t *a, const int16_t *b, int16_t *out, int n) {
    for (int i = 0; i < n; i++)
        out[i] = (a[i] < b[i]) ? a[i] : b[i];
}
