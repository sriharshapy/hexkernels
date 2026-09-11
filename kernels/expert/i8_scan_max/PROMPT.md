# Task: i8_scan_max

Implement an inclusive prefix maximum scan over a 1D int8 array.

## Signature
```c
void candidate_kernel(const int8_t *in, int8_t *out, int n);
```

## Semantics
```
out[0] = in[0]
out[i] = max(out[i-1], in[i])   for i in [1, n)
```
Equivalently: `out[i] = max(in[0], in[1], ..., in[i])`.
The running maximum never decreases.

## HVX hints
Like prefix sum, prefix max has a sequential dependency. Use a two-pass strategy:
1. **Pass 1 (vectorized per chunk):** For each 128-element chunk, compute the local
   prefix max within the vector using log2(128)=7 rounds of `Q6_Vb_vmax_VbVb`
   with shifted/rotated copies. Record the chunk maximum.
2. **Pass 2 (scalar carry):** Apply the carry-in maximum from the previous chunk
   using `Q6_Vb_vmax_VbVb` with a broadcast.

Key intrinsic: `Q6_Vb_vmax_VbVb` (element-wise int8 max).

## Notes
- Output is int8 (same type as input).
- A kernel that broadcasts the global max to every output element will fail
  on the early outputs where the running max is still below the global max.
- n=1000; handle the 104-element tail.
