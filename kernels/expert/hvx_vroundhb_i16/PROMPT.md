# HVX Drill: Round-and-Narrow int16 -> int8 (n=210 pairs)

Implement ONLY this function (Hexagon HVX C):
```c
void candidate_kernel(const int16_t *a, const int16_t *b, int8_t *out, int n);
```

**Semantics:** `a` and `b` each have `n` int16 elements; `out` has `2*n`
int8 elements, INTERLEAVED:
```
out[2*i]   = sat8( round_div256(b[i]) )
out[2*i+1] = sat8( round_div256(a[i]) )
```
where `round_div256(x) = (x + 128) >> 8` (arithmetic shift — floor
division by 256 after adding the round-half-up bias, so ties round toward
`+infinity`: `128 -> 1`, `-128 -> 0`), and `sat8` clamps to `[-128, 127]`.
This matches the HVX intrinsic `Q6_Vb_vround_VhVh_sat(Vu=a, Vv=b)` — note
`b` (the second/`Vv` argument) lands at the EVEN output positions and `a`
at the ODD ones.

- `n=210` (not a multiple of the 64-halfword HVX vector width) — handle
  the scalar tail.
- Arrays are 128-byte aligned (`HVX_ALIGN`).

Include `<hexagon_types.h>` and `<hexagon_protos.h>`. Do NOT write main().
Respond with a single complete C code block.
