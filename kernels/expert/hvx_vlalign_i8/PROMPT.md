# HVX Drill: Cross-Block Aligned Window Extraction, vlalign (n=768, Rt=19)

Implement ONLY this function (Hexagon HVX C):
```c
void candidate_kernel(const int8_t *a, const int8_t *b, int8_t *out, int n);
```

Semantics: n=768 = 6 full 128-byte blocks. Fix RT=19. For each block
index k, define combined[0:128) = b's block k, combined[128:256) = a's
block k (same {Vv:Vu} ordering as valign). For i in [0, 128):
```
out[k*128 + i] = combined[128 - RT + i]
```
Concretely: out[k*128+i] = a[k*128+i-RT] when i >= RT, otherwise
b[k*128 + (128-RT+i)].

Use the HVX intrinsic Q6_V_vlalign_VVR(Vu, Vv, Rt) with Vu = a's block,
Vv = b's block, Rt = 19: vlalign shifts the OTHER direction from valign
-- it returns the 128-byte window ending RT bytes before the top of the
conceptual 256-byte vector {Vv : Vu}. This op is block-granular (defined
only over a pair of full 128-byte vectors), so n is a multiple of 128
and there is no scalar byte-tail to handle.

- n=768 (exactly 6 vector pairs; no tail). Rt=19 is a fixed compile-time
  constant, not a runtime input.
- Arrays are 128-byte aligned (HVX_ALIGN).

Include <hexagon_types.h> and <hexagon_protos.h>. Do NOT write main().
Respond with a single complete C code block.
