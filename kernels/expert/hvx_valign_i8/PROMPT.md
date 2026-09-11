# HVX Drill: Cross-Block Aligned Window Extraction, valign (n=1024, Rt=37)

Implement ONLY this function (Hexagon HVX C):
```c
void candidate_kernel(const int8_t *a, const int8_t *b, int8_t *out, int n);
```

Semantics: n=1024 = 8 full 128-byte blocks. Fix RT=37. For each block
index k, define combined[0:128) = b's block k, combined[128:256) = a's
block k (b first, then a -- NOT a-then-b). For i in [0, 128):
```
out[k*128 + i] = combined[RT + i]
```
Concretely: out[k*128+i] = b[k*128+RT+i] when RT+i < 128, otherwise
a[k*128 + (RT+i-128)].

Use the HVX intrinsic Q6_V_valign_VVR(Vu, Vv, Rt) with Vu = a's block,
Vv = b's block, Rt = 37: it returns the 128-byte window starting at
offset Rt into the conceptual 256-byte vector {Vv : Vu} (Vv contributes
the low addresses, Vu the high addresses). This op is block-granular
(defined only over a pair of full 128-byte vectors), so n is a multiple
of 128 and there is no scalar byte-tail to handle.

- n=1024 (exactly 8 vector pairs; no tail). Rt=37 is a fixed compile-time
  constant, not a runtime input.
- Arrays are 128-byte aligned (HVX_ALIGN).

Include <hexagon_types.h> and <hexagon_protos.h>. Do NOT write main().
Respond with a single complete C code block.
