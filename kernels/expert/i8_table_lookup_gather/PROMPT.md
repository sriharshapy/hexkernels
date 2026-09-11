# Task: word table lookup / gather

Gather `n_idx` int32 values from a large table using a runtime index array:

```
out[i] = table[idx[i]]     for i in [0, n_idx)
```

- `table`   : `table_words = 16384` int32 values (64 KB, larger than L1)
- `idx`     : `n_idx = 32768` indices in `[0, table_words)` (random)
- `out`     : `n_idx` int32 results

Random indices into a table larger than L1 make a scalar gather miss to DDR on
almost every lookup. The Hexagon HVX hardware gather fetches 32 elements per
instruction from a **VTCM-resident** region. Stage the table in VTCM once, then
gather:

```c
Q6_vgather_ARMVw(vtcm_dst, vtcm_table_base, region_len, byte_offset_vector);
/* a dummy read of vtcm_dst stalls until the gather completes */
```

The VTCM identity translation is already installed by the harness (the historical
`vgather` fault 0x26 was VTCM-not-mapped).

Implement:

```c
void candidate_kernel(const int32_t *table, const int32_t *idx, int32_t *out,
                      int table_words, int n_idx);
```
