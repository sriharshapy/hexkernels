#include <stdint.h>
void candidate_kernel(const int32_t *a, const int32_t *bias, int8_t *out, int n,
                      int32_t mult, int shift, int8_t zp) {
    for (int i = 0; i < n; i++) {
        int64_t sum = (int64_t)a[i] + (int64_t)bias[i];
        int64_t v   = sum * (int64_t)mult;
        int64_t half = shift > 0 ? ((int64_t)1 << (shift - 1)) : 0;
        int64_t r   = (v >= 0) ? ((v + half) >> shift) : -(((-v) + half) >> shift);
        r += zp;
        if (r > 127)  r = 127;
        if (r < -128) r = -128;
        out[i] = (int8_t)r;
    }
}
