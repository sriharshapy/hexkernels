/* Near-miss: ignores the circular roll, just adds a+b directly.
   Fails whenever k != 0 (which the sweep always includes non-zero k). */
#include <stdint.h>
void candidate_kernel(const int8_t *a, const int8_t *b, int8_t *out, int n, int k) {
    (void)k;
    for (int i = 0; i < n; i++)
        out[i] = (int8_t)(a[i] + b[i]);
}
