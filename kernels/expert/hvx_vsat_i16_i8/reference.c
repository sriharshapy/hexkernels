#include <stdint.h>
/* Scalar baseline: chunked saturating narrow, concatenated (b then a). */
static int8_t sat8(int v) {
    if (v > 127) return 127;
    if (v < -128) return -128;
    return (int8_t)v;
}
void candidate_kernel(const int16_t *a, const int16_t *b, int8_t *out, int n) {
    int base = 0;
    while (base < n) {
        int m = (n - base < 64) ? (n - base) : 64;
        for (int j = 0; j < m; j++) {
            out[2*base + j]     = sat8(b[base + j]);
            out[2*base + m + j] = sat8(a[base + j]);
        }
        base += m;
    }
}
