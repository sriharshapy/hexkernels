# HVX Drill: Deinterleaved Zero-Extend uint8->uint16, vzxt (n=384)

Implement ONLY this function (Hexagon HVX C):
```c
void candidate_kernel(const uint8_t *a, uint16_t *out, int n);
```

Semantics: n=384 = 3 full 128-byte input blocks. Output has n uint16
elements (widening doubles the byte width per element, not the element
count). For each block k (128 bytes at offset k*128 in `a`,
128 uint16 at offset k*128 in `out`), and for j in [0, 64):
```
out[k*128 + j]      = (uint16_t)a[k*128 + 2*j]       // even-indexed bytes
out[k*128 + 64 + j] = (uint16_t)a[k*128 + 2*j + 1]   // odd-indexed bytes
```
This is NOT a plain sequential widen -- the output is deinterleaved
into an even-byte half followed by an odd-byte half, because that is
exactly the native layout the HVX widening instruction produces.

Use `Q6_Wuh_vzxt_Vub(Vu)`, which returns a VectorPair whose low half
(`Q6_V_lo_W`) holds the even-indexed input bytes zero-extended to
uint16, and whose high half (`Q6_V_hi_W`) holds the odd-indexed bytes.
This op is block-granular, so n is a multiple of 128 and there is no
scalar byte-tail.

- n=384 (exactly 3 blocks; no tail).
- Arrays are 128-byte aligned (HVX_ALIGN).

Include <hexagon_types.h> and <hexagon_protos.h>. Do NOT write main().
Respond with a single complete C code block.
