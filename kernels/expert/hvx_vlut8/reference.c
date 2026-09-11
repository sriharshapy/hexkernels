#include <stdint.h>
/* Scalar baseline: 256-entry byte lookup, unsigned-reinterpreted index. */
void candidate_kernel(const int8_t *a, int8_t *out, int n, const int8_t *lut) {
    for (int i = 0; i < n; i++) {
        uint8_t idx = (uint8_t)a[i];
        out[i] = lut[idx];
    }
}
