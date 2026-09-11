/* Near-miss: hardcodes pos=0 — always writes at position 0 regardless of
 * the runtime `pos` argument. Fails for positions 31, 32, 63. */
#include <stdint.h>
void candidate_kernel(int8_t *k_cache, int8_t *v_cache,
                      const int8_t *k_new, const int8_t *v_new,
                      int pos, int num_heads, int cache_len, int head_dim) {
    (void)pos;  /* BUG: pos ignored */
    for (int h = 0; h < num_heads; h++) {
        /* BUG: hardcoded pos=0 */
        int cache_base = h * cache_len * head_dim;  /* = (h*cache_len + 0) * head_dim */
        int new_base   = h * head_dim;
        for (int d = 0; d < head_dim; d++) {
            k_cache[cache_base + d] = k_new[new_base + d];
            v_cache[cache_base + d] = v_new[new_base + d];
        }
    }
}
