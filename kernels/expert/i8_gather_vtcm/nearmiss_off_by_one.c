/* NEARMISS: off-by-one row index (table[idx[i]+1] instead of table[idx[i]],
 * wrapping at vocab_size). Plausible but wrong; must score INCORRECT. */
#include <stdint.h>
void candidate_kernel(const int8_t *table, const int32_t *idx, int8_t *out,
                      int T, int E, int vocab_size) {
    for (int i = 0; i < T; i++) {
        int32_t row = idx[i] + 1;
        if (row >= vocab_size) row = 0;
        const int8_t *src = table + (int64_t)row * E;
        int8_t       *dst = out   + (int64_t)i * E;
        for (int e = 0; e < E; e++) dst[e] = src[e];
    }
}
