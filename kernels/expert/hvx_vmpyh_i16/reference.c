#include <stdint.h>
/* Scalar baseline: Q15 fractional multiply-high, round+saturate. */
void candidate_kernel(const int16_t *a, const int16_t *b, int16_t *out, int n) {
    for (int i = 0; i < n; i++) {
        int32_t P = (int32_t)a[i] * (int32_t)b[i];
        int32_t bias = (P >= 0) ? 16384 : -16384;
        int32_t q = (P + bias) / 32768;
        if (q > 32767) q = 32767;
        if (q < -32768) q = -32768;
        out[i] = (int16_t)q;
    }
}
