# HVX Drill: 4-Wide SAD Against a Fixed Pattern, vrsad (g=98 groups)

Implement ONLY this function (Hexagon HVX C):
```c
void candidate_kernel(const uint8_t *a, uint32_t *out, int g);
```

Semantics: fixed pattern = {10, 20, 30, 40} (unsigned bytes). For k in
[0, g): out[k] = sum_{j=0..3} |(int)a[4*k+j] - (int)pattern[j]|.

Use Q6_Wuw_vrsad_WubRubI(Vuu, Rt, Iu1) where:
- Vuu is a HVX_VectorPair built via Q6_W_vcombine_VV(Vu, Vu) -- duplicate
  your single 128-byte input vector into both halves of the pair (this
  intrinsic is normally used for block-matching against two independent
  128B regions, but duplicating the same vector lets you use it for a
  single-vector-vs-fixed-pattern SAD).
- Rt = 10 | (20<<8) | (30<<16) | (40<<24) (== 0x281E140A) -- the
  pattern's 4 bytes packed little-endian, byte k of Rt == pattern[k].
- Iu1 = 0.
Then Q6_V_lo_W of the resulting pair gives the 32 uint32 SAD values for
that 128-byte (32-group) input vector.

- g=98 (not a multiple of 32 groups) -- handle the scalar tail.
- Arrays are 128-byte aligned (HVX_ALIGN).

Include <hexagon_types.h> and <hexagon_protos.h>. Do NOT write main().
Respond with a single complete C code block.
