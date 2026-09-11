#include <stdint.h>
/*
 * KV-cache update baseline — pure indexed copy, no arithmetic.
 *
 * For each head h in [0, num_heads):
 *   k_cache[(h * cache_len + pos) * head_dim + d] = k_new[h * head_dim + d]
 *   v_cache[(h * cache_len + pos) * head_dim + d] = v_new[h * head_dim + d]
 *   for d in [0, head_dim).
 *
 * Pinned dimensions: num_heads=4, head_dim=32, cache_len=64.
 */
void candidate_kernel(int8_t *k_cache, int8_t *v_cache,
                      const int8_t *k_new, const int8_t *v_new,
                      int pos, int num_heads, int cache_len, int head_dim) {
    for (int h = 0; h < num_heads; h++) {
        int cache_base = (h * cache_len + pos) * head_dim;
        int new_base   = h * head_dim;
        for (int d = 0; d < head_dim; d++) {
            k_cache[cache_base + d] = k_new[new_base + d];
            v_cache[cache_base + d] = v_new[new_base + d];
        }
    }
}
