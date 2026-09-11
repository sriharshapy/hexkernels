/* Near-miss: skips the scale/64 requantisation of the index, using
 * idx = clamp(x + 128, 0, 255) directly. Correct only when scale == 64; here
 * scale != 64 so the index (and thus the looked-up value) differs on most inputs. */
#include <stdint.h>
void candidate_kernel(const int8_t *in, int8_t *out, int n, const int8_t *lut, int8_t scale) {
    (void)scale;
    for (int i = 0; i < n; i++) {
        int idx = (int)in[i] + 128;        /* MISSING: * scale / 64 */
        if (idx < 0) idx = 0; if (idx > 255) idx = 255;
        out[i] = lut[idx];
    }
}
