#ifndef KERNEL_API_H
#define KERNEL_API_H
#include <stdint.h>
/*
 * KV-cache update: write new K and V vectors into pre-allocated cache buffers
 * at a given sequence position (indexed write).
 *
 * Operation:
 *   For each head h in [0, num_heads):
 *     k_cache[h * cache_len * head_dim + pos * head_dim + d] = k_new[h * head_dim + d]
 *     v_cache[h * cache_len * head_dim + pos * head_dim + d] = v_new[h * head_dim + d]
 *   for d in [0, head_dim).
 *
 * Equivalently: for head h, copy head_dim bytes from k_new/v_new row h into
 *               k_cache/v_cache row (h * cache_len + pos).
 *
 * Pinned dimensions: num_heads=4, head_dim=32, cache_len=64.
 *   k_cache, v_cache: [num_heads * cache_len * head_dim] = [8192] int8 each.
 *   k_new, v_new:     [num_heads * head_dim] = [128] int8 each.
 *   pos:              int, in [0, cache_len).
 *
 * This is a pure indexed copy — no arithmetic, no normalization.
 */
void candidate_kernel(int8_t *k_cache, int8_t *v_cache,
                      const int8_t *k_new, const int8_t *v_new,
                      int pos, int num_heads, int cache_len, int head_dim);
#endif /* KERNEL_API_H */
