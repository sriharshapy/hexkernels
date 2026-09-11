/* Near-miss: uses wrong cache offset formula — swaps head and pos dimensions.
 * Writes to (pos * num_heads + h) * head_dim instead of (h * cache_len + pos) * head_dim.
 * Produces garbage for pos > 0 or when num_heads != cache_len. */
#include <stdint.h>
void candidate_kernel(int8_t *k_cache, int8_t *v_cache,
                      const int8_t *k_new, const int8_t *v_new,
                      int pos, int num_heads, int cache_len, int head_dim) {
    for (int h = 0; h < num_heads; h++) {
        /* BUG: wrong stride — pos and head axes swapped */
        int cache_base = (pos * num_heads + h) * head_dim;
        int new_base   = h * head_dim;
        for (int d = 0; d < head_dim; d++) {
            k_cache[cache_base + d] = k_new[new_base + d];
            v_cache[cache_base + d] = v_new[new_base + d];
        }
    }
}
