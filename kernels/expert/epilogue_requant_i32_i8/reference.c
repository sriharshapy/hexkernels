#include <stdint.h>
void candidate_kernel(const int32_t *acc, int8_t *out, int n, int32_t mult, int shift) {
    for (int i = 0; i < n; i++) {
        int64_t v = (int64_t)acc[i] * (int64_t)mult;
        int64_t half = shift > 0 ? ((int64_t)1 << (shift - 1)) : 0;
        int64_t r = (v >= 0) ? ((v + half) >> shift) : -(((-v) + half) >> shift);
        if (r > 127) r = 127;
        if (r < -128) r = -128;
        out[i] = (int8_t)r;
    }
}
