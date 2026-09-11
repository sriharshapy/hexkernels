#include <stdint.h>
/* Scalar baseline: elementwise unsigned absolute difference. */
void candidate_kernel(const uint8_t *a, const uint8_t *b, uint8_t *out, int n) {
    for (int i = 0; i < n; i++) {
        int d = (int)a[i] - (int)b[i];
        out[i] = (uint8_t)(d < 0 ? -d : d);
    }
}
