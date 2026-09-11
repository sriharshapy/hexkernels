# HVX Drill: Elementwise int8 Bitwise XOR (n=1021)

Implement ONLY this function (Hexagon HVX C):
```c
void candidate_kernel(const int8_t *a, const int8_t *b, int8_t *out, int n);
```

**Semantics:** `out[i] = (int8_t)((uint8_t)a[i] ^ (uint8_t)b[i])` for `i` in `[0, n)`.

Raw bit-pattern XOR, sign-agnostic. Use the HVX intrinsic `Q6_V_vxor_VV` (NOT
`vand`/`vor`). For example `0xFF ^ 0x0F = 0xF0` and `x ^ x = 0`.

- `n=1021` (not a multiple of the 128-byte HVX vector width) — handle the scalar tail.
- Arrays are 128-byte aligned (`HVX_ALIGN`).

Include `<hexagon_types.h>` and `<hexagon_protos.h>`. Do NOT write main().
Respond with a single complete C code block.
