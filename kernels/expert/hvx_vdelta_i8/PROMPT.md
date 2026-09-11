# HVX Drill: Per-Block Byte Reversal via vdelta (n=896)

Implement ONLY this function (Hexagon HVX C):
```c
void candidate_kernel(const int8_t *a, int8_t *out, int n);
```

Semantics: n=896 = 7 full 128-byte blocks. For each block b (starting at
byte offset b*128), and for i in [0, 128):
```
out[b*128 + i] = a[b*128 + 127 - i]
```
i.e. reverse the byte order within each 128-byte block.

Use the HVX general permutation intrinsic Q6_V_vdelta_VV(Vu, Vctrl) with
the control vector Vctrl set to all bytes = 0xFF (build it with
Q6_Vb_vsplat_R(0xFF)) -- this specific control pattern drives every stage
of the delta-swap network to swap, producing a full 128-byte reversal of
the input vector. This op is block-granular (defined only over a full
128-byte vector), so n is a multiple of 128 and there is no scalar
byte-tail to handle.

- n=896 (exactly 7 vectors; no tail).
- Arrays are 128-byte aligned (HVX_ALIGN).

Include <hexagon_types.h> and <hexagon_protos.h>. Do NOT write main().
Respond with a single complete C code block.
