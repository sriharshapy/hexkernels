/* Scalar baseline: uint8 alpha blend.
 * out[i] = (uint8_t)((alpha*a[i] + (256-alpha)*b[i] + 128) >> 8). */
#include <stdint.h>
void candidate_kernel(const uint8_t *a, const uint8_t *b, uint8_t *out,
                      int n, uint16_t alpha) {
    for (int i = 0; i < n; i++) {
        int v = ((int)alpha*(int)a[i] + (256-(int)alpha)*(int)b[i] + 128) >> 8;
        out[i] = (uint8_t)v;
    }
}
