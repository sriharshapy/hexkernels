/* NEARMISS: off-by-one index (table[idx[i]+1]). Plausible but wrong; produces a
 * complete (wrong) output, must score INCORRECT. */
#include <stdint.h>
void candidate_kernel(const int16_t *table, const int16_t *idx, int16_t *out,
                      int table_hwords, int n_idx) {
    for (int i = 0; i < n_idx; i++) {
        int j = idx[i] + 1; if (j >= table_hwords) j = 0;
        out[i] = table[j];
    }
}
