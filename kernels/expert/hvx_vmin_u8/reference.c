#include <stdint.h>
/* Scalar baseline: elementwise unsigned min, uint8. */
void candidate_kernel(const uint8_t *a, const uint8_t *b, uint8_t *out, int n) {
    for (int i = 0; i < n; i++)
        out[i] = (a[i] < b[i]) ? a[i] : b[i];
}
