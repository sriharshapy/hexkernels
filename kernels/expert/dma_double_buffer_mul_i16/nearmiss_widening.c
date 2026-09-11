/* Near-miss: saturating narrow instead of a truncating (wraparound) narrow.
 * On overflow this clamps to +/-32768/32767 instead of wrapping the low 16
 * bits, e.g. 32767*32767 should truncate-wrap but here it saturates. */
#include <stdint.h>
void candidate_kernel(const int16_t *a, const int16_t *b, int16_t *out, int n) {
    for (int i = 0; i < n; i++) {
        int32_t v = (int32_t)a[i] * (int32_t)b[i];
        if (v > 32767) v = 32767;
        if (v < -32768) v = -32768;
        out[i] = (int16_t)v;
    }
}
