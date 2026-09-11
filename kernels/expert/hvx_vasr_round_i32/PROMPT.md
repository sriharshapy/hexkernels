# HVX Drill: Rounded+Saturated int32->int16 Shift (n=200 pairs)

Implement ONLY this function (Hexagon HVX C):
```c
void candidate_kernel(const int32_t *lo, const int32_t *hi, int16_t *out, int n, int shift);
```

`lo` and `hi` each have `n` int32 elements. `out` has `2*n` int16 elements,
**interleaved**:
```
out[2*i]   = sat16( round_shift(lo[i], shift) )
out[2*i+1] = sat16( round_shift(hi[i], shift) )
```
where `round_shift(x, s) = (s == 0) ? x : (x + (1 << (s-1))) >> s` (arithmetic
shift — round-half-up, then floor-divide by `2^s`), and `sat16` clamps to
`[-32768, 32767]`.

This matches the HVX intrinsic `Q6_Vh_vasr_VwVwR_rnd_sat(Vu=hi_vector,
Vv=lo_vector, shift)` — note the interleave: `Vv`'s narrowed lanes land at
**even** output positions, `Vu`'s at **odd** output positions. Do NOT confuse
this with the non-rounding sibling `Q6_Vh_vasr_VwVwR_sat` (plain truncating
shift, no `+1<<(s-1)` bias) — for example `lo=16, shift=5` rounds to `1`
(truncating would give `0`).

- `n=200` (not a multiple of the 32-lane HVX int32 vector width) — handle the scalar tail.
- Arrays are 128-byte aligned (`HVX_ALIGN`).

Include `<hexagon_types.h>` and `<hexagon_protos.h>`. Do NOT write main().
Respond with a single complete C code block.
