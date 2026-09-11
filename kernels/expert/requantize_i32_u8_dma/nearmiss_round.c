/* Near-miss: arithmetic-shift truncation (round toward -inf) instead of
 * round-half-away-from-zero. Matches on many inputs but differs whenever the
 * discarded fraction is >= 0.5. */
#include <stdint.h>
void candidate_kernel(const int32_t *a, uint8_t *out, int n,
                      int32_t mult, int shift, uint8_t zp) {
    for (int i = 0; i < n; i++) {
        int64_t v = (int64_t)a[i] * (int64_t)mult;
        int64_t r = v >> shift;          /* arithmetic shift, no round-half-away */
        r += (int64_t)(uint64_t)zp; if (r > 255) r = 255; if (r < 0) r = 0;
        out[i] = (uint8_t)r;
    }
}
