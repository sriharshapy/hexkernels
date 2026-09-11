# Task: embedding-style D-wide row gather, VTCM-staged

Gather `T` rows of width `E` bytes from a table using a runtime index array:

```
out[i, :] = table[idx[i], :] for i in [0, T)
out[i*E + e] = table[idx[i]*E + e] for e in [0, E)
```

- `table` : `vocab_size x E` int8, row-major (E=128, a multiple of 128 = one HVX vector).
 The table is small enough to stage in VTCM but larger than L1, so a naive per-row DDR
 read misses cache on almost every random row.
- `idx` : `T` int32 indices in `[0, vocab_size)` (random; T is large).
- `out` : `T x E` int8, row-major.

To go fast: stage the WHOLE table into VTCM ONCE with a single uDMA transfer, then for
each row do a fast 128-byte HVX vector load from the VTCM-resident table row and a vector
store to the output row -- instead of reading each row cold from DDR. VTCM is identity-
mapped at 0xd8400000. uDMA: build a static/global Type-0 descriptor
`{next, ctrl=len, src, dst}`, then `Q6_dmstart_A(&desc)` / `Q6_R_dmwait()` (a stack
descriptor no-ops at -O2; the VTCM identity translation is already installed by the
harness).

Implement ONLY this function (Hexagon HVX C):

```c
void candidate_kernel(const int8_t *table, const int32_t *idx, int8_t *out,
 int T, int E, int vocab_size);
```

single complete C code block and CLOSE the fence with ```.
