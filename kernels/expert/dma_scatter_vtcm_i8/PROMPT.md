# Task: scatter via HVX hardware scatter + VTCM staging

Scatter `n` int8 values to positions given by a runtime index array:

```
out[idx[i]] = (int32_t) values[i]     for i in [0, n)
```

- `values` : `n` int8 values (widened to int32 for the scatter)
- `idx`    : `n` indices that form a PERMUTATION of `[0, n)` (every destination
             position 0..n-1 is written by EXACTLY one `i`, so no destination
             address collides within any group of lanes -- this is what makes the
             operation well-defined regardless of internal processing/lane order).
- `out`    : `n` int32 slots (word-granularity; the HVX scatter instruction writes
             whole words, not individual bytes)

`n` is large (table > L1), so scattering one DDR word at a time (the scalar
baseline) pays a DDR-miss-like cost on almost every write. The achievability bar
scatters directly into a VTCM-resident staging table using the HVX hardware scatter,
then DMAs the completed table out to `out` in one bulk transfer:

```c
Q6_vscatter_RMVwV(vtcm_table_base /* Word32 */, region_len_minus_1 /* Word32 */,
                  byte_offset_vector /* HVX_Vector: idx[i]*4 per lane */,
                  data_vector /* HVX_Vector: widened values[i] per lane */);
```

The VTCM identity translation (`add_translation`) is already installed by the
harness at 0xd8400000. `n` is not necessarily a multiple of 32 (the scatter width);
handle any remainder with scalar stores directly into the VTCM table (or DDR, for the
final tail).

Implement ONLY this function (Hexagon HVX C):
```c
void candidate_kernel(const int8_t *values, const int32_t *idx, int32_t *out, int n);
```
Include <hexagon_types.h> and <hexagon_protos.h>. Do NOT write main(). Respond with a
single complete C code block and CLOSE the fence with ```.
