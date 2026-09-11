#include <stdint.h>
/* Scalar baseline: negative average (a-b)>>1, floor, no saturation. */
void candidate_kernel(const int8_t *a, const int8_t *b, int8_t *out, int n) {
    for (int i = 0; i < n; i++)
        out[i] = (int8_t)(((int)a[i] - (int)b[i]) >> 1);
}
