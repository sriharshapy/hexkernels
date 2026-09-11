#include <stdint.h>
/* Scalar baseline: round-and-narrow int16 -> int8 (b at even, a at odd). */
static int8_t sat8(int v) {
    if (v > 127) return 127;
    if (v < -128) return -128;
    return (int8_t)v;
}
static int round_div256(int16_t x) {
    return ((int)x + 128) >> 8;
}
void candidate_kernel(const int16_t *a, const int16_t *b, int8_t *out, int n) {
    for (int i = 0; i < n; i++) {
        out[2*i]     = sat8(round_div256(b[i]));
        out[2*i + 1] = sat8(round_div256(a[i]));
    }
}
