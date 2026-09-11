# HVX Drill: Elementwise Unsigned Absolute Difference, uint8 (n=1069)

Implement ONLY this function (Hexagon HVX C):
```c
void candidate_kernel(const uint8_t *a, const uint8_t *b, uint8_t *out, int n);
```

**Semantics:** `out[i] = (uint8_t)abs((int)a[i] - (int)b[i])` for `i` in
`[0, n)`, treating `a[i]` and `b[i]` as **unsigned** bytes (not
reinterpreted as signed) before differencing in wider precision. This
matches the HVX intrinsic `Q6_Vub_vabsdiff_VubVub`. For example
`|0 - 255| = 255` — reinterpreting as signed bytes first would (wrongly)
give `|0 - (-1)| = 1`.

- `n=1069` (not a multiple of the 128-byte HVX vector width) — handle the
  scalar tail.
- Arrays are 128-byte aligned (`HVX_ALIGN`).

Include `<hexagon_types.h>` and `<hexagon_protos.h>`. Do NOT write main().
Respond with a single complete C code block.
