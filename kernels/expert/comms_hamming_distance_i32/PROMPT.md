# Bitwise Hamming Distance per int32 word pair (n=1024)

Implement:
```c
void candidate_kernel(const int32_t *a, const int32_t *b, int32_t *out, int n);
```

Semantics:
- `out[i] = popcount( a[i] ^ b[i] )` — the number of bit positions where `a[i]` and `b[i]` differ
- Result range: 0..32 (0 means identical words, 32 means all bits differ)
- ALL 32 bits are counted, including the sign/high bit
- Equivalent: `out[i] = (int32_t)__builtin_popcount((uint32_t)(a[i] ^ b[i]))`
- n=1024; handle tail (n may not be a multiple of 128)

HVX hint: XOR a and b with `Q6_V_vxor_VV`, then apply a vectorized popcount
(e.g. `Q6_Vb_vpopcount_Vb` on byte lanes + `Q6_Vh_vdmpy_VhRh` reduce,
or vrmpyub-based Hamming-weight).
The correctness bar: all 32 bits of each XOR word must be counted.
