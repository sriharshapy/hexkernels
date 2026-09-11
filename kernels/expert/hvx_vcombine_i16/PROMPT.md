# HVX Drill: Per-Block int16 Concatenation via vcombine (g=3 blocks)

Implement ONLY this function (Hexagon HVX C):
```c
void candidate_kernel(const int16_t *a, const int16_t *b, int16_t *out, int g);
```

Semantics: g=3 blocks; each of a, b has 64*g int16 elements. For each
block k (64 elements at offset k*64), and for i in [0, 64):
```
out[k*128 + i]      = a[k*64 + i]
out[k*128 + 64 + i] = b[k*64 + i]
```
i.e. out is simply the natural per-block concatenation [a_block, b_block].

The catch: the HVX intrinsic is `Q6_W_vcombine_VV(Vu, Vv)`, which builds a
256-byte VectorPair whose LOW half (extract with `Q6_V_lo_W`) equals its
SECOND argument `Vv`, and whose HIGH half (`Q6_V_hi_W`) equals its FIRST
argument `Vu` -- the opposite of what you might expect. To get the
natural concatenation out=[a,b], call `Q6_W_vcombine_VV(b_block, a_block)`
(b first, a second) and store `Q6_V_lo_W(pair)` (== a_block) to the first
64 elements, `Q6_V_hi_W(pair)` (== b_block) to the next 64.

This op is block-granular (defined only over a pair of full 64-element
/ 128-byte int16 vectors), so g is chosen so both inputs are exact
multiples of 64 elements, and there is no scalar tail.

- g=3 (3 full vector-pair blocks; no tail).
- Arrays are 128-byte aligned (HVX_ALIGN).

Include <hexagon_types.h> and <hexagon_protos.h>. Do NOT write main().
Respond with a single complete C code block.
