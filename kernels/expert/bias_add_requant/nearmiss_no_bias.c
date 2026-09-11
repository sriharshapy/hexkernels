/* Near-miss: ignores bias[] — requantizes a[] alone without the bias term.
   Fails whenever bias[i] != 0, which is virtually all inputs. */
#include <stdint.h>
void candidate_kernel(const int32_t *a, const int32_t *bias, int8_t *out, int n,
                      int32_t mult, int shift, int8_t zp) {
    (void)bias;
    for (int i = 0; i < n; i++) {
        long long v   = (long long)a[i] * (long long)mult;
        long long h   = shift > 0 ? (1LL << (shift - 1)) : 0;
        long long r   = (v >= 0) ? ((v + h) >> shift) : -(((-v) + h) >> shift);
        r += zp;
        if (r > 127)  r = 127;
        if (r < -128) r = -128;
        out[i] = (signed char)r;
    }
}
