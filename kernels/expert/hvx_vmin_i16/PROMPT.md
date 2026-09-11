# HVX Drill: Elementwise int16 Min (n=700)

Implement ONLY this function (Hexagon HVX C):
```c
void candidate_kernel(const int16_t *a, const int16_t *b, int16_t *out, int n);
```

**Semantics:** `out[i] = (a[i] < b[i]) ? a[i] : b[i]` for `i` in `[0, n)`, using
**SIGNED** int16 comparison. This matches `Q6_Vh_vmin_VhVh` (NOT the
unsigned-halfword `Q6_Vuh_vmin_VuhVuh`). For example with `a=-1, b=1`: the signed
min is `-1` (unsigned-halfword comparison would wrongly treat `-1` as `65535` and
return `1`).

- `n=700` (not a multiple of the 64-lane HVX int16 vector width) — handle the scalar tail.
- Arrays are 128-byte aligned (`HVX_ALIGN`).

Include `<hexagon_types.h>` and `<hexagon_protos.h>`. Do NOT write main().
Respond with a single complete C code block.
