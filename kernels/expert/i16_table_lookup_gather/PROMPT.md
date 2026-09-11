# Task: int16 table lookup / gather

Gather `n_idx` int16 values from a large table using a runtime index array:

```
out[i] = table[idx[i]]     for i in [0, n_idx)
```

- `table`      : `table_hwords = 16384` int16 values (32 KB, larger than L1)
- `idx`        : `n_idx = 32768` indices in `[0, table_hwords)` (random)
- `out`        : `n_idx` int16 results

The Hexagon HVX halfword hardware gather fetches 64 halfwords per instruction
from a **VTCM-resident** region using a vector of halfword byte-offsets. Stage
the table in VTCM once, then gather:

```c
Q6_vgather_ARMVh(vtcm_dst, vtcm_table_base, region_len, halfword_offset_vector);
/* offsets are BYTE offsets (idx * 2); a dummy read of vtcm_dst syncs */
```

The scalar baseline misses to DDR on every random lookup. The VTCM identity
translation is installed by the harness.

Implement:

```c
void candidate_kernel(const int16_t *table, const int16_t *idx, int16_t *out,
                      int table_hwords, int n_idx);
```
