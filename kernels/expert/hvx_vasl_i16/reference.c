#include <stdint.h>
/* Scalar baseline: arithmetic (wrapping) shift-left, int16. */
void candidate_kernel(const int16_t *a, int16_t *out, int n, int shift) {
    for (int i = 0; i < n; i++)
        out[i] = (int16_t)((uint16_t)a[i] << shift);
}
