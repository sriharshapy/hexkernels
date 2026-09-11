/* Near-miss: arithmetic-shift truncation (round toward -inf) instead of
 * round-half-away-from-zero. Matches on many inputs but differs whenever the
 * discarded fraction is >= 0.5 (positive) or on negative rounding. */
#include <stdint.h>
void candidate_kernel(const int32_t *a, int8_t *out, int n,
                      int32_t mult, int shift, int8_t zp) {
    for (int i = 0; i < n; i++) {
        int64_t v = (int64_t)a[i] * (int64_t)mult;
        int64_t r = v >> shift;          /* arithmetic shift, no round-half-away */
        r += zp; if (r > 127) r = 127; if (r < -128) r = -128;
        out[i] = (int8_t)r;
    }
}
