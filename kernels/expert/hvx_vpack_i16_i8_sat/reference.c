#include <stdint.h>
static int8_t sat8(int x) {
    if (x > 127) return 127;
    if (x < -128) return -128;
    return (int8_t)x;
}
/* Scalar baseline: per-block saturating pack, out=[sat(b),sat(a)]. */
void candidate_kernel(const int16_t *a, const int16_t *b, int8_t *out, int g) {
    for (int k = 0; k < g; k++) {
        for (int i = 0; i < 64; i++) {
            out[k*128 + i]      = sat8(b[k*64 + i]);
            out[k*128 + 64 + i] = sat8(a[k*64 + i]);
        }
    }
}
