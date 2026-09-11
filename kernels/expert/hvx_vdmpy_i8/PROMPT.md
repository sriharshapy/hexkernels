# HVX Drill: Dual uint8 x int8 Multiply-Accumulate (g=262 groups)

Implement ONLY this function (Hexagon HVX C):
```c
void candidate_kernel(const uint8_t *a, const int8_t w[4], int16_t *out, int g);
```

`a` has `4*g` uint8 elements; group `k` occupies `a[4*k .. 4*k+4)`. `w` is a FIXED
4-byte signed weight, shared across every group (like the HVX scalar-register
operand of `Q6_Vh_vdmpy_VubRb`). `out` has `2*g` int16 elements.

**Semantics:** for `k` in `[0, g)`:
```
out[2*k]   = (int16_t)((int)a[4*k+0]*w[0] + (int)a[4*k+1]*w[1])
out[2*k+1] = (int16_t)((int)a[4*k+2]*w[2] + (int)a[4*k+3]*w[3])
```
`a` is unsigned uint8, `w` is signed int8. Two's-complement **WRAP** on overflow
(the MAC does not saturate) — matches `Q6_Vh_vdmpy_VubRb` with `Rt` packed
little-endian as `(w[3]<<24)|(w[2]<<16)|(w[1]<<8)|w[0]`.

- `g=262` groups (not a multiple of the 32-group vdmpy width: one 128-byte input
  vector produces 64 int16 outputs) — handle the scalar tail.
- Arrays are 128-byte aligned (`HVX_ALIGN`).

Include `<hexagon_types.h>` and `<hexagon_protos.h>`. Do NOT write main().
Respond with a single complete C code block.
