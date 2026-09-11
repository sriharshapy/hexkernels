/* Scalar baseline: SIGNED int8 elementwise min.
 * out[i] = a[i] < b[i] ? a[i] : b[i] (signed comparison). */
#include <stdint.h>
void candidate_kernel(const int8_t *a, const int8_t *b, int8_t *out, int n) {
    for (int i = 0; i < n; i++)
        out[i] = (a[i] < b[i]) ? a[i] : b[i];
}
