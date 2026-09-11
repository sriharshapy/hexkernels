# HVX Drill: Per-Block Byte Shuffle / Interleave (n=640)

Implement ONLY this function (Hexagon HVX C):
```c
void candidate_kernel(const int8_t *a, int8_t *out, int n);
```

Semantics: n=640 = 5 full 128-byte blocks. For each block b (starting at
byte offset b*128), and for i in [0, 64):
```
out[b*128 + 2*i]     = a[b*128 + i]
out[b*128 + 2*i + 1] = a[b*128 + 64 + i]
```
i.e. within each 128-byte block, interleave the low half (bytes 0..63)
and the high half (bytes 64..127) of the block.

Use the HVX intrinsic Q6_Vb_vshuff_Vb(Vu) -- a single-operand instruction
that performs exactly this in-vector interleave. This op is block-granular
(defined only over a full 128-byte vector), so n is a multiple of 128 and
there is no scalar byte-tail to handle.

- n=640 (exactly 5 vectors; no tail).
- Arrays are 128-byte aligned (HVX_ALIGN).

Include <hexagon_types.h> and <hexagon_protos.h>. Do NOT write main().
Respond with a single complete C code block.
