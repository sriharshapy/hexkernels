#include <stdint.h>
/* Scalar baseline: per-block natural concatenation [a_block, b_block]. */
void candidate_kernel(const int16_t *a, const int16_t *b, int16_t *out, int g) {
    for (int k = 0; k < g; k++) {
        for (int i = 0; i < 64; i++) {
            out[k*128 + i]      = a[k*64 + i];
            out[k*128 + 64 + i] = b[k*64 + i];
        }
    }
}
