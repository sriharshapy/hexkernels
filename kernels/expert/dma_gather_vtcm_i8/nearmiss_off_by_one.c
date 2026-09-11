/* Near-miss: off-by-one index (reads table[idx[i]+1] instead of
 * table[idx[i]]), a plausible transcription bug. */
#include <stdint.h>
void candidate_kernel(const int32_t *table, const int32_t *idx, int8_t *out,
                      int table_words, int n_idx) {
    for (int i = 0; i < n_idx; i++) {
        int j = idx[i] + 1;
        if (j >= table_words) j = table_words - 1;
        out[i] = (int8_t)table[j];
    }
}
