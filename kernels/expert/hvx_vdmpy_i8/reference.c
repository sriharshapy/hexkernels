#include <stdint.h>
/* Scalar baseline: dual unsigned-byte x signed-byte MAC to int16, wrapping. */
void candidate_kernel(const uint8_t *a, const int8_t w[4], int16_t *out, int g) {
    for (int k = 0; k < g; k++) {
        int s0 = (int)a[4*k+0]*w[0] + (int)a[4*k+1]*w[1];
        int s1 = (int)a[4*k+2]*w[2] + (int)a[4*k+3]*w[3];
        out[2*k]   = (int16_t)s0;
        out[2*k+1] = (int16_t)s1;
    }
}
