# HVX Drill: Elementwise int8 Bitwise AND (n=1029)

Implement ONLY this function (Hexagon HVX C):
```c
void candidate_kernel(const int8_t *a, const int8_t *b, int8_t *out, int n);
```

**Semantics:** `out[i] = (int8_t)((uint8_t)a[i] & (uint8_t)b[i])` for `i` in `[0, n)`
— a plain bitwise AND of the byte patterns (sign-agnostic). Matches
`Q6_V_vand_VV` (NOT `Q6_V_vor_VV`).

- `n=1029` (not a multiple of the 128-byte HVX vector width) — handle the scalar tail.
- Arrays are 128-byte aligned (`HVX_ALIGN`).

Include `<hexagon_types.h>` and `<hexagon_protos.h>`. Do NOT write main().
Respond with a single complete C code block.
