# fp16 FLASH-ATTENTION tile (tiled online-softmax recurrence, HVX)

Implement
`candidate_kernel(const hvx_hf *Q, const hvx_hf *K, const hvx_hf *V, hvx_hf *O, int SQ, int SK, int DH)`
(`hvx_hf` = `__fp16`) computing the flash-attention output for a Q tile of
`SQ=16` query rows against `SK=64` key/value rows of head dim `DH=64`, by
STREAMING the K/V rows in tiles of `FLASH_TILE_K=16` (4 tiles) and maintaining
the standard **online-softmax recurrence**:

```
scale = FLASH_SCALE (0.125)
for each query row i in [0,SQ):
 m = -INFINITY; l = 0; acc[0..DH-1] = 0
 for each K/V tile t of FLASH_TILE_K rows:
 for k in tile: s[k] = scale * dot(Q[i,:], K[t*FLASH_TILE_K+k,:])
 tile_max = max_k s[k]
 new_m = max(m, tile_max)
 corr = exp(m - new_m) <- rescale factor for the OLD state
 l = l * corr
 acc[:] = acc[:] * corr
 for k in tile:
 p = exp(s[k] - new_m)
 l += p
 acc[:] += p * V[t*FLASH_TILE_K+k, :]
 m = new_m
 O[i,:] = acc[:] / l
```

This is mathematically **identical** to standard (full-softmax, non-tiled)
attention — `softmax(Q.K^T * scale) . V` — the online recurrence is an
algorithmic reformulation that never materializes the full `SQ x SK` score
matrix and rescales the running accumulator/sum whenever a later tile raises
the running max. The single most important correctness rule: whenever
`new_m > m` (the running max increases), you MUST rescale BOTH `l` and every
element of `acc` by `corr = exp(m - new_m)` BEFORE folding in the new tile's
contribution — forgetting this is the classic flash-attention bug.

`DH=64` is exactly one 64-lane fp16 HVX vector (no padding). Output is
compared against a float32 standard-attention reference with `hvx_close_f16bits`
tolerance (HVX float arithmetic goes through the non-IEEE qf16 path and float
ops reorder, so this is NOT bit-exact).

`the HVX headers`. Do NOT write `main`. Respond with a single complete C
code block.
