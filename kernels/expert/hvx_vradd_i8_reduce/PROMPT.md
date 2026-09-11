# HVX Drill: 4-Wide int8 Reduce-Add to int32 (g=106 groups)

Implement ONLY this function (Hexagon HVX C):
```c
void candidate_kernel(const int8_t *a, int32_t *out, int g);
```

Semantics: for k in [0, g): `out[k] = sum_{j=0..3} (int32)a[4*k+j]`
(signed int8 sum of each 4-element group, widened to int32).

Use `Q6_Vw_vrmpy_VbVb(Vu, Vones)` where `Vones` is a vector with every
byte lane = int8 value 1 (build it with `Q6_Vb_vsplat_R(1)`): a signed
4-wide dot product against all-1s is exactly a 4-wide sum. One 128-byte
input vector (32 groups) produces one 128-byte (32 int32) output vector.

- g=106 (not a multiple of 32 groups) -- handle the scalar tail.
- Arrays are 128-byte aligned (HVX_ALIGN).

Include <hexagon_types.h> and <hexagon_protos.h>. Do NOT write main().
Respond with a single complete C code block.
