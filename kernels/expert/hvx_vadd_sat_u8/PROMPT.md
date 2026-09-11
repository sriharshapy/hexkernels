# HVX Drill: Elementwise Saturating uint8 Add (n=777)

Implement ONLY this function (Hexagon HVX C):
```c
void candidate_kernel(const uint8_t *a, const uint8_t *b, uint8_t *out, int n);
```

**Semantics:** `out[i] = sat_u8((int)a[i] + (int)b[i])` for `i` in `[0, n)`, where
`sat_u8(x)` clamps `x` to `[0, 255]`.

Use **saturation**, not two's-complement wrap — this matches the HVX intrinsic
`Q6_Vub_vadd_VubVub_sat` (NOT the wrapping `Q6_Vb_vadd_VbVb`). For example
`200 + 200 = 400` saturates to `255`, not `144`.

- `n=777` (not a multiple of the 128-byte HVX vector width) — handle the scalar tail.
- Arrays are 128-byte aligned (`HVX_ALIGN`).

Include `<hexagon_types.h>` and `<hexagon_protos.h>`. Do NOT write main().
Respond with a single complete C code block.
