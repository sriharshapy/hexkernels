#include <stdint.h>
/* Scalar baseline: saturating elementwise int16 subtract. */
static int16_t sat16(int v) {
    if (v > 32767) return 32767;
    if (v < -32768) return -32768;
    return (int16_t)v;
}
void candidate_kernel(const int16_t *a, const int16_t *b, int16_t *out, int n) {
    for (int i = 0; i < n; i++)
        out[i] = sat16((int)a[i] - (int)b[i]);
}
