#include <stdint.h>
/* Scalar baseline: bitwise NOT. */
void candidate_kernel(const int8_t *a, int8_t *out, int n) {
    for (int i = 0; i < n; i++)
        out[i] = (int8_t)(~a[i]);
}
