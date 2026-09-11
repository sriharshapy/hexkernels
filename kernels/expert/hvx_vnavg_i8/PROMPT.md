# HVX Drill: Negative Average, int8 (n=1091)

Implement ONLY this function (Hexagon HVX C):
```c
void candidate_kernel(const int8_t *a, const int8_t *b, int8_t *out, int n);
```

**Semantics:** `out[i] = (int8_t)(((int)a[i] - (int)b[i]) >> 1)` for `i` in
`[0, n)`. The shift is computed in wider (`int`) precision, then
narrowed — an ARITHMETIC (floor) shift, with NO rounding bias and NO
saturation. This matches the HVX intrinsic `Q6_Vb_vnavg_VbVb`.

Note the name: "negative average" here means `(a-b)/2` floored, NOT
`-(a+b)/2` — e.g. `a=10, b=4` gives `3`, not `-7`.

- `n=1091` (not a multiple of the 128-byte HVX vector width) — handle the
  scalar tail.
- Arrays are 128-byte aligned (`HVX_ALIGN`).

Include `<hexagon_types.h>` and `<hexagon_protos.h>`. Do NOT write main().
Respond with a single complete C code block.
