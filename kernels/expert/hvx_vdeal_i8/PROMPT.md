# HVX Drill: Per-Block Byte Deal / De-interleave (n=768)

Implement ONLY this function (Hexagon HVX C):
```c
void candidate_kernel(const int8_t *a, int8_t *out, int n);
```

Semantics: n=768 = 6 full 128-byte blocks. For each block b (starting at
byte offset b*128), and for i in [0, 64):
```
out[b*128 + i]      = a[b*128 + 2*i]
out[b*128 + 64 + i] = a[b*128 + 2*i + 1]
```
i.e. within each 128-byte block, gather all even-indexed bytes into the
low half of the output block and all odd-indexed bytes into the high
half. This is the exact inverse of the shuffle/interleave operation.

Use the HVX intrinsic Q6_Vb_vdeal_Vb(Vu) -- a single-operand instruction
that performs exactly this in-vector de-interleave. This op is
block-granular (defined only over a full 128-byte vector), so n is a
multiple of 128 and there is no scalar byte-tail to handle.

- n=768 (exactly 6 vectors; no tail).
- Arrays are 128-byte aligned (HVX_ALIGN).

Include <hexagon_types.h> and <hexagon_protos.h>. Do NOT write main().
Respond with a single complete C code block.
