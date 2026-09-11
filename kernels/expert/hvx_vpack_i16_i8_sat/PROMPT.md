# HVX Drill: Per-Block int16->int8 Saturating Pack, vpack (g=2 blocks)

Implement ONLY this function (Hexagon HVX C):
```c
void candidate_kernel(const int16_t *a, const int16_t *b, int8_t *out, int g);
```

Semantics: define sat8(x) = clamp(x, -128, 127). g=2 blocks; each of a, b
has 64*g=128 int16 elements. For each block k (64 int16 elements at
offset k*64 in a/b, 128 int8 at offset k*128 in out), and for i in [0, 64):
```
out[k*128 + i]      = sat8(b[k*64 + i])
out[k*128 + 64 + i] = sat8(a[k*64 + i])
```
NOTE the operand order: the first 64 output bytes come from `b` (the
SECOND function argument), the last 64 from `a` (the FIRST argument).

Use `Q6_Vb_vpack_VhVh_sat(Vu, Vv)`: its low output half is `sat(Vv)`
and its high output half is `sat(Vu)` -- so call it with `Vu = a_block`,
`Vv = b_block` to match the semantics above. This op is block-granular
(defined only over a pair of full 64-element / 128-byte int16 vectors),
so g is chosen so both inputs are exact multiples of 64 elements, and
there is no scalar tail.

- g=2 (2 full vector-pair blocks; no tail).
- Arrays are 128-byte aligned (HVX_ALIGN).

Include <hexagon_types.h> and <hexagon_protos.h>. Do NOT write main().
Respond with a single complete C code block.
