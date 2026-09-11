# HVX Drill: Broadcast/Splat a Scalar to a Vector (n=1011)

Implement ONLY this function (Hexagon HVX C):
```c
void candidate_kernel(int8_t val, int8_t *out, int n);
```

Semantics: out[i] = val for i in [0, n) -- broadcast the single scalar
byte val into every output element.

Use the HVX intrinsic Q6_Vb_vsplat_R(Rt), which broadcasts the low 8 bits
of a 32-bit scalar register into every byte lane of a 128-byte vector.
Pass val widened to an int (sign bits above bit 7 are ignored by the
intrinsic, which only reads the low byte).

- n=1011 (not a multiple of the 128-byte HVX vector width) - handle the scalar tail.
- The output array is 128-byte aligned (HVX_ALIGN).

Include <hexagon_types.h> and <hexagon_protos.h>. Do NOT write main().
Respond with a single complete C code block.
