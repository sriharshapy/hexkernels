# HVX Drill: Elementwise int8 Max (n=1050)

Implement ONLY this function (Hexagon HVX C):
```c
void candidate_kernel(const int8_t *a, const int8_t *b, int8_t *out, int n);
```

**Semantics:** `out[i] = (a[i] > b[i]) ? a[i] : b[i]` for `i` in `[0, n)`, using
**SIGNED** int8 comparison. This matches `Q6_Vb_vmax_VbVb` (NOT the unsigned-byte
`Q6_Vub_vmax_VubVub`). For example with `a=-1, b=1`: the signed max is `1`
(unsigned-byte comparison would wrongly treat `-1` as `255` and return `-1`).

- `n=1050` (not a multiple of the 128-byte HVX vector width) — handle the scalar tail.
- Arrays are 128-byte aligned (`HVX_ALIGN`).

Include `<hexagon_types.h>` and `<hexagon_protos.h>`. Do NOT write main().
Respond with a single complete C code block.
