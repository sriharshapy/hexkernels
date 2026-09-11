/* Near-miss: ignores idx and reads in_a[i] (identity gather) instead of
 * in_a[idx[i]].  The add in_b[i] is correct, but the gathered element is
 * wrong for any non-identity index pattern.  The harness uses two distinct
 * random index sweeps, so this fails on both.
 */
#include <stdint.h>
void candidate_kernel(const int8_t *in_a, const int8_t *in_b,
                      const int32_t *idx, int8_t *out, int n) {
    (void)idx;  /* BUG: ignores the gather index array */
    for (int i = 0; i < n; i++) {
        /* BUG: reads in_a[i] instead of in_a[idx[i]] */
        out[i] = (int8_t)((int16_t)in_a[i] + (int16_t)in_b[i]);
    }
}
