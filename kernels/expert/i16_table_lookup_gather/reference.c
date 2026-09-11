/* Baseline: scalar int16 table lookup. Random indices into a table larger than
 * L1 miss to DDR on every access. Speedup denominator. */
#include <stdint.h>
void candidate_kernel(const int16_t *table, const int16_t *idx, int16_t *out,
                      int table_hwords, int n_idx) {
    (void)table_hwords;
    for (int i = 0; i < n_idx; i++) out[i] = table[idx[i]];
}
