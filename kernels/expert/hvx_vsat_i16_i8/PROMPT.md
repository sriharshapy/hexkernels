# HVX Drill: Saturating Narrow int16 -> int8 (n=270)

Implement ONLY this function (Hexagon HVX C):
```c
void candidate_kernel(const int16_t *a, const int16_t *b, int8_t *out, int n);
```

**Semantics:** `a` and `b` each have `n` int16 elements; `out` has `2*n`
int8 elements. Processing happens in CHUNKS of up to 64 elements
(matching one HVX vector of int16). For a chunk covering source index
range `[base, base+m)` (`m=64` for full chunks, `m=n-base` for the final
partial chunk) and local index `j` in `[0, m)`:
```
out[2*base + j]     = sat8(b[base + j])
out[2*base + m + j] = sat8(a[base + j])
```
i.e. within each chunk, `b`'s saturated values come FIRST, `a`'s come
SECOND — this is a CONCATENATION, not an interleave. `sat8` clamps to
`[-128, 127]`. This matches the HVX intrinsic
`Q6_Vb_vpack_VhVh_sat(Vu=a, Vv=b)`.

- `n=270` (not a multiple of 64) — the final chunk has `m=14`.
- Arrays are 128-byte aligned (`HVX_ALIGN`).

Include `<hexagon_types.h>` and `<hexagon_protos.h>`. Do NOT write main().
Respond with a single complete C code block.
