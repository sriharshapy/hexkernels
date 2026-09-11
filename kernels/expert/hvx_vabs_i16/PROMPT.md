# HVX Drill: Elementwise int16 Absolute Value (n=513)

Implement ONLY this function (Hexagon HVX C):
```c
void candidate_kernel(const int16_t *a, int16_t *out, int n);
```

**Semantics:** `out[i] = abs(a[i])` for `i` in `[0, n)`, **except** the two's-complement
corner case `a[i] == -32768`: its magnitude (32768) doesn't fit in `int16_t`, so the
result **wraps**: `abs(-32768) = -32768` (NOT `32767`). This matches the HVX intrinsic
`Q6_Vh_vabs_Vh` (NOT the saturating `Q6_Vh_vabs_Vh_sat`, which would clamp to `32767`).

- `n=513` (not a multiple of the 64-lane HVX int16 vector width) — handle the scalar tail.
- Arrays are 128-byte aligned (`HVX_ALIGN`).

Include `<hexagon_types.h>` and `<hexagon_protos.h>`. Do NOT write main().
Respond with a single complete C code block.
