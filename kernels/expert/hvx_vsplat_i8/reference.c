#include <stdint.h>
/* Scalar baseline: broadcast. */
void candidate_kernel(int8_t val, int8_t *out, int n) {
    for (int i = 0; i < n; i++)
        out[i] = val;
}
