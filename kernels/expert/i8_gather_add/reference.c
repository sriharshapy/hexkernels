#include <stdint.h>
/*
 * i8_gather_add baseline (scalar reference implementation).
 * out[i] = (int8_t)((int16_t)in_a[idx[i]] + (int16_t)in_b[i])
 * for i in [0, n). Two's-complement int8 wraparound arithmetic.
 */
void candidate_kernel(const int8_t *in_a, const int8_t *in_b,
                      const int32_t *idx, int8_t *out, int n) {
    for (int i = 0; i < n; i++) {
        out[i] = (int8_t)((int16_t)in_a[idx[i]] + (int16_t)in_b[i]);
    }
}
