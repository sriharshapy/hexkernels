/* Near-miss: wraparound add instead of saturating add. Values near 255
 * wrap around (e.g. 250+50=300 -> 44) instead of clamping to 255. */
#include <stdint.h>
void candidate_kernel(const uint8_t *a, uint8_t *out, int n) {
    for (int i = 0; i < n; i++) out[i] = (uint8_t)(a[i] + 50);
}
