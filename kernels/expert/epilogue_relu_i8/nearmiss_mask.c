/* Near-miss: implements "relu" as x[i] & 0x7F -- this only clears the sign
 * bit instead of zeroing negatives, e.g. -1 (0xFF) & 0x7F = 0x7F = 127
 * (very wrong; should be 0). Compiles, plausible-looking bitwise "trick",
 * fails whenever x[i] < 0 (which the input distribution guarantees). */
#include <stdint.h>
void candidate_kernel(const int8_t *x, int8_t *out, int n) {
    for (int i = 0; i < n; i++) out[i] = (int8_t)((uint8_t)x[i] & 0x7F);
}
