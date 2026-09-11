/* Baseline: scalar table lookup + narrow. Random indices into a table
 * larger than L1 miss to DDR on almost every access. This is the speedup
 * denominator. */
#include <stdint.h>
void candidate_kernel(const int32_t *table, const int32_t *idx, int8_t *out,
                      int table_words, int n_idx) {
    (void)table_words;
    for (int i = 0; i < n_idx; i++) out[i] = (int8_t)table[idx[i]];
}
