# HVX Drill: Elementwise int8 Bitwise NOT (n=1017)

Implement ONLY this function (Hexagon HVX C):
```c
void candidate_kernel(const int8_t *a, int8_t *out, int n);
```

Semantics: out[i] = (int8_t)(~a[i]) for i in [0, n) (ones-complement
bit-flip, e.g. ~0 = -1, ~(-1) = 0, ~0x0F = 0xF0).

Use the unary HVX intrinsic Q6_V_vnot_V (single vector operand - this op
has no second input).

- n=1017 (not a multiple of the 128-byte HVX vector width) - handle the scalar tail.
- Arrays are 128-byte aligned (HVX_ALIGN).

Include <hexagon_types.h> and <hexagon_protos.h>. Do NOT write main().
Respond with a single complete C code block.
