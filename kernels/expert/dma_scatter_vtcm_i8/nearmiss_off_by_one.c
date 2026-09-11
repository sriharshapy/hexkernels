/* Near-miss: off-by-one destination index (writes to idx[i]+1 mod n
 * instead of idx[i]), a plausible transcription bug. */
#include <stdint.h>
void candidate_kernel(const int8_t *values, const int32_t *idx, int32_t *out, int n) {
    for (int i = 0; i < n; i++) out[(idx[i] + 1) % n] = (int32_t)values[i];
}
