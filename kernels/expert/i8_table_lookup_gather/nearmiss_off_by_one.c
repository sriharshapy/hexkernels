/* NEARMISS: off-by-one index (table[idx[i]+1]). Plausible but wrong; must score
 * INCORRECT. */
#include <stdint.h>
void candidate_kernel(const int32_t *table, const int32_t *idx, int32_t *out,
                      int table_words, int n_idx) {
    for (int i = 0; i < n_idx; i++) {
        int j = idx[i] + 1; if (j >= table_words) j = 0;
        out[i] = table[j];
    }
}
