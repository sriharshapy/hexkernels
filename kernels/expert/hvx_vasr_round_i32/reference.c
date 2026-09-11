#include <stdint.h>
/* Scalar baseline: rounded+saturated arithmetic shift, int32 -> int16. */
static int16_t round_shift_sat(int32_t x, int shift) {
    int32_t r = (shift == 0) ? x : ((x + (1 << (shift - 1))) >> shift);
    if (r > 32767) r = 32767;
    if (r < -32768) r = -32768;
    return (int16_t)r;
}

void candidate_kernel(const int32_t *lo, const int32_t *hi, int16_t *out, int n, int shift) {
    for (int i = 0; i < n; i++) {
        out[2*i]   = round_shift_sat(lo[i], shift);
        out[2*i+1] = round_shift_sat(hi[i], shift);
    }
}
