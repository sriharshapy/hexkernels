# HVX Drill: Elementwise int8 Add (n=1037)

Implement ONLY this function (Hexagon HVX C):
```c
void candidate_kernel(const int8_t *a, const int8_t *b, int8_t *out, int n);
```

**Semantics:** `out[i] = (int8_t)((int)a[i] + (int)b[i])` for `i` in `[0, n)`.

Use two's-complement **wraparound** on overflow — this matches the HVX intrinsic
`Q6_Vb_vadd_VbVb` (NOT the saturating `Q6_Vb_vadd_VbVb_sat`). For example
`127 + 127` wraps to `-2`, not `127`.

- `n=1037` (not a multiple of the 128-byte HVX vector width) — handle the scalar tail.
- Arrays are 128-byte aligned (`HVX_ALIGN`).

Include `<hexagon_types.h>` and `<hexagon_protos.h>`. Do NOT write main().
Respond with a single complete C code block.
