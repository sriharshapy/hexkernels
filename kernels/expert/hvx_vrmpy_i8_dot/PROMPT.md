# HVX Drill: 4-Wide int8 Dot-Product Accumulate (g=261 groups)

Implement ONLY this function (Hexagon HVX C):
```c
void candidate_kernel(const int8_t *a, const int8_t *b, int32_t *out, int g);
```

`a` and `b` each have `4*g` int8 elements; group `k` occupies indices `[4*k, 4*k+4)`.

**Semantics:** for `k` in `[0, g)`:
```
out[k] = sum_{j=0..3} (int32_t)a[4*k+j] * (int32_t)b[4*k+j]
```
Both `a` and `b` are **SIGNED** int8 (matches `Q6_Vw_vrmpy_VbVb`, NOT the
mixed-sign `Q6_Vw_vrmpy_VubVb`, which treats the first operand as unsigned byte).
No saturation is needed: the magnitude of each sum is well within `int32` range.

- `g=261` groups (not a multiple of the 32-group HVX vrmpy width: one 128-byte
  input vector produces 32 groups' worth of int32 output) — handle the scalar tail.
- Arrays are 128-byte aligned (`HVX_ALIGN`).

Include `<hexagon_types.h>` and `<hexagon_protos.h>`. Do NOT write main().
Respond with a single complete C code block.
