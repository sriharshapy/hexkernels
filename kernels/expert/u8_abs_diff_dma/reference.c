/* Scalar baseline: uint8 absolute difference.
 * out[i] = (a[i] > b[i]) ? a[i]-b[i] : b[i]-a[i]. */
#include <stdint.h>
void candidate_kernel(const uint8_t *a, const uint8_t *b, uint8_t *out, int n) {
    for (int i = 0; i < n; i++)
        out[i] = (uint8_t)(a[i] > b[i] ? a[i] - b[i] : b[i] - a[i]);
}
