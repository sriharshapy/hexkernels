/* Near-miss: gathers correctly (uses idx) but skips requantization.
 * Simply copies table[idx[i]] to out[i] without applying mult/shift/zp.
 * Fails on any param set where mult*table[idx[i]] >> shift + zp != table[idx[i]],
 * i.e., almost all cases with mult != 1 or shift != 0 or zp != 0.
 */
#include <stdint.h>
void candidate_kernel(const int8_t *table, const int32_t *idx, int8_t *out, int n,
                      int32_t mult, int shift, int8_t zp) {
    (void)mult; (void)shift; (void)zp;  /* BUG: ignores requant params */
    for (int i = 0; i < n; i++) {
        out[i] = table[idx[i]];  /* BUG: no requantization */
    }
}
