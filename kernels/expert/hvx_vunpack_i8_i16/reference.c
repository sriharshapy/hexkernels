#include <stdint.h>
/* Scalar baseline: sign-extend int8 -> int16. */
void candidate_kernel(const int8_t *a, int16_t *out, int n) {
    for (int i = 0; i < n; i++)
        out[i] = (int16_t)(int8_t)a[i];
}
