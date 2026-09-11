# HVX Drill: Elementwise uint8 Bitwise OR (n=1033)

Implement ONLY this function (Hexagon HVX C):
```c
void candidate_kernel(const uint8_t *a, const uint8_t *b, uint8_t *out, int n);
```

**Semantics:** `out[i] = (uint8_t)(a[i] | b[i])` for `i` in `[0, n)`.

Use the HVX intrinsic `Q6_V_vor_VV` (a raw bit-pattern OR - dtype-agnostic,
NOT vand/vxor). For example 0xF0 | 0x0F = 0xFF and 0x00 | x = x.

- n=1033 (not a multiple of the 128-byte HVX vector width) - handle the scalar tail.
- Arrays are 128-byte aligned (HVX_ALIGN).

Include <hexagon_types.h> and <hexagon_protos.h>. Do NOT write main().
Respond with a single complete C code block.
