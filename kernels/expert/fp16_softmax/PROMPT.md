# fp16 row-wise softmax on HVX (partial vectorization: exp is unavoidably scalar)

Implement `candidate_kernel(const __fp16 *x, __fp16 *out, int R, int C)` for
`R == 8`, `C == 256` (4 HVX vectors of 64 fp16 lanes per row, exact tiling).
For each row `r`:

```
rowmax    = max_j x[r][j]
e[r][j]   = exp((float)x[r][j] - rowmax)
rowsum    = sum_j e[r][j]
out[r][j] = (float)e[r][j] / rowsum
```

Output is compared to a float32 scalar reference with an fp16 tolerance --
HVX float arithmetic is non-IEEE (qf16), so bit-exactness is NOT required.

- **Max-reduce (vectorized):** running per-lane max across the C/64 blocks
  with `Q6_Vhf_vmax_VhfVhf` (native at v68), then unpack the 64-lane result
  ONCE per row and finish the reduction with 64 scalar compares
  (negligible).
- **exp (unavoidably scalar):** HVX has NO vector transcendental, so this
  must be a per-element scalar loop (`expf`) -- fuse it with the running
  sum accumulation so you don't need a second pass over the row.
- **Normalize (vectorized):** broadcast `1/rowsum` into a vector via its
  `__fp16` bit pattern and `Q6_Vh_vsplat_R`, then multiply each of the C/64
  blocks in qf16 (`Q6_Vqf16_vmpy_VhfVhf`) instead of a per-element scalar
  divide.
- The denominator is a plain scalar (no-HVX) fp16 implementation of the
  same three steps; your HVX expert must beat it by >=1.2x kernel-cycles --
  the win comes from vectorizing the max-reduce and normalize, since the
  exp cost itself is paid identically by both.
