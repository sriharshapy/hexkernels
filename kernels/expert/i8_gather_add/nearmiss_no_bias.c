/* Near-miss: performs gather correctly but omits the bias addition.
 * out[i] = in_a[idx[i]] with no in_b term.
 * Fails whenever in_b[i] != 0, which is true for most positions with random data.
 */
#include <stdint.h>
void candidate_kernel(const int8_t *in_a, const int8_t *in_b,
                      const int32_t *idx, int8_t *out, int n) {
    (void)in_b;  /* BUG: ignores the bias array entirely */
    for (int i = 0; i < n; i++) {
        out[i] = in_a[idx[i]];  /* BUG: missing + in_b[i] */
    }
}
