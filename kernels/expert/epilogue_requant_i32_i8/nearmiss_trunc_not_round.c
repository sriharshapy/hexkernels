/* Near-miss: plain truncating arithmetic right shift, no "+half" rounding term
 * (uses int64 correctly for the multiply, but skips round-half-away-from-zero).
 * Off-by-up-to-1 on many elements whenever shift>0 and the true remainder is
 * nonzero (e.g. the harness's mult=200,shift=8 sweep entry). */
#include <stdint.h>
void candidate_kernel(const int32_t *acc, int8_t *out, int n, int32_t mult, int shift) {
    for (int i = 0; i < n; i++) {
        int64_t v = (int64_t)acc[i] * (int64_t)mult;
        int64_t r = v >> shift;   /* no rounding */
        if (r > 127)  r = 127;
        if (r < -128) r = -128;
        out[i] = (int8_t)r;
    }
}
