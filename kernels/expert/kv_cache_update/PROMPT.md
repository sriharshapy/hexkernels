# Task: kv_cache_update

Implement `candidate_kernel` in C for Hexagon HVX (128-byte vectors).

## Signature
```c
void candidate_kernel(int8_t *k_cache, int8_t *v_cache,
                      const int8_t *k_new, const int8_t *v_new,
                      int pos, int num_heads, int cache_len, int head_dim);
```

## Semantics
KV-cache indexed write: append new Key and Value vectors into pre-allocated cache buffers
at sequence position `pos`. Pure copy — no arithmetic, no normalization.

**Operation:**
```
For each head h in [0, num_heads):
  dest_k = k_cache + (h * cache_len + pos) * head_dim
  dest_v = v_cache + (h * cache_len + pos) * head_dim
  src_k  = k_new  + h * head_dim
  src_v  = v_new  + h * head_dim

  Copy head_dim bytes: dest_k[d] = src_k[d], dest_v[d] = src_v[d]
```

Only the `pos`-th slot of each head should be written. All other cache slots must remain unchanged.

**Pinned dimensions:** `num_heads=4, head_dim=32, cache_len=64`.
- `k_cache`, `v_cache`: `[num_heads * cache_len * head_dim] = [8192]` int8 each.
- `k_new`, `v_new`: `[num_heads * head_dim] = [128]` int8 each.
- `pos`: runtime, in `[0, cache_len)` — do NOT hardcode.

## HVX guidance
- Use `HVX_Vector` (128B); `k_new` and `v_new` are exactly 128 bytes (one HVX vector each).
- For each head, load one 32-byte slice of k_new/v_new and store to the cache slot.
- `head_dim=32` bytes = 1/4 HVX vector; can batch 4 heads in 1 HVX load/store if heads are contiguous.
- All arrays are 128-byte aligned.

## Constraints
- `pos` is a runtime argument — do NOT hardcode it.
- Only write the designated slot; do NOT corrupt other cache positions.
- No arithmetic — pure indexed copy.
- No float.
