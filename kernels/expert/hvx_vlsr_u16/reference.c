#include <stdint.h>
/* Scalar baseline: logical (zero-fill) shift-right, uint16. */
void candidate_kernel(const uint16_t *a, uint16_t *out, int n, int shift) {
    for (int i = 0; i < n; i++)
        out[i] = (uint16_t)(a[i] >> shift);
}
