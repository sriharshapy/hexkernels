# Task: narrow (int8-output) table lookup via HVX hardware gather

Gather `n_idx` values from a large int32 table and narrow each to int8:

```
out[i] = (int8_t) table[idx[i]]     for i in [0, n_idx)
```

- `table`      : `table_words` int32 values (each in the int8 range -128..127),
                 large enough to exceed L1, so a scalar gather misses to DDR on
                 almost every lookup.
- `idx`        : `n_idx` indices in `[0, table_words)` (random)
- `out`        : `n_idx` int8 results (each table value narrowed to int8)

Stage the whole table into VTCM once via uDMA, then use the HVX hardware gather to
fetch 32 table words per instruction from the VTCM-resident table using a vector of
byte offsets:

```c
Q6_vgather_ARMVw(vtcm_dst /* HVX_Vector* */, vtcm_table_base /* Word32 */,
                 region_len_minus_1 /* Word32 */, byte_offset_vector /* HVX_Vector */);
/* a dummy (volatile) read of vtcm_dst stalls until the gather completes */
```

Narrow each gathered int32 lane to int8 (e.g. pack/shuffle the low byte of each lane,
or extract scalarly) when writing to `out`. The VTCM identity translation
(`add_translation`) is already installed by the harness at 0xd8400000. `n_idx` is not
necessarily a multiple of 32 (the gather width); handle any remainder.

Implement ONLY this function (Hexagon HVX C):
```c
void candidate_kernel(const int32_t *table, const int32_t *idx, int8_t *out,
                      int table_words, int n_idx);
```
Include <hexagon_types.h> and <hexagon_protos.h>. Do NOT write main(). Respond with a
single complete C code block and CLOSE the fence with ```.
