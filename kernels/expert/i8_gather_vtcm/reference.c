/* Baseline: scalar per-byte row gather straight from DDR. Random row indices
 * into a table larger than L1 miss to DDR on almost every row. This is the
 * speedup denominator. */
#include <stdint.h>
void candidate_kernel(const int8_t *table, const int32_t *idx, int8_t *out,
                      int T, int E, int vocab_size) {
    (void)vocab_size;
    for (int i = 0; i < T; i++) {
        const int8_t *src = table + (int64_t)idx[i] * E;
        int8_t       *dst = out   + (int64_t)i * E;
        for (int e = 0; e < E; e++) dst[e] = src[e];
    }
}
