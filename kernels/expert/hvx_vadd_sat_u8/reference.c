#include <stdint.h>
/* Scalar baseline: saturating uint8 add, clamp to [0,255]. */
void candidate_kernel(const uint8_t *a, const uint8_t *b, uint8_t *out, int n) {
    for (int i = 0; i < n; i++) {
        int r = (int)a[i] + (int)b[i];
        if (r > 255) r = 255;
        out[i] = (uint8_t)r;
    }
}
