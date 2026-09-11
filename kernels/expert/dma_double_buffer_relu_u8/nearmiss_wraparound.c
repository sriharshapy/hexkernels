/* Near-miss: plain wraparound subtract instead of saturating (floors at 0)
 * subtract. a[i]=0 gives 0-128 = -128 which wraps to 128 as uint8 instead
 * of clamping to 0. */
#include <stdint.h>
void candidate_kernel(const uint8_t *a, uint8_t *out, int n) {
    for (int i = 0; i < n; i++) out[i] = (uint8_t)(a[i] - 128);
}
