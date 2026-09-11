#include <stdint.h>
/* Scalar baseline: 4-wide signed reduce-add. */
void candidate_kernel(const int8_t *a, int32_t *out, int g) {
    for (int k = 0; k < g; k++) {
        int32_t sum = 0;
        for (int j = 0; j < 4; j++)
            sum += (int32_t)a[4*k+j];
        out[k] = sum;
    }
}
