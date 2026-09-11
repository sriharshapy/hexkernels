# fp16 32x32x32 matmul + residual add on HVX (pure vector, no matrix engine)

Implement `candidate_kernel(const __fp16 *A, const __fp16 *B, const __fp16 *C,
__fp16 *out, int n, int k_dim)` for `n == k_dim == 32`:

```
acc[i][j] = sum_k A[i*k_dim+k] * B[k*n+j] (float32-accumulate, M=N=K=32)
m[i][j] = (__fp16)acc[i][j] (fp16-round)
out[i][j] = (float)m[i][j] + (float)C[i*n+j] (elementwise residual, n x n)
```

- Output is compared to the float32-accumulate scalar reference with an fp16
 tolerance -- HVX float arithmetic is non-IEEE (qf16), so bit-exactness is
 NOT required.
- Use a row-broadcast/AXPY qf16 matmul over the K=32 reduction
 (`Q6_Vqf16_vmpy_VhfVhf` accumulated with `Q6_Vqf16_vadd_Vqf16Vqf16`).
- Fuse the residual add as a native HVX qf16 vector op
 (`Q6_Vqf16_vadd_VhfVhf`, since v68 lacks a native `Vhf+Vhf` add) directly on the
 matmul result vector, THEN convert once. Read the matching row of `C` via
 a vector load (bulk-copy `C` into a padded, zero-tailed buffer first) --
 do NOT repack `C` element-by-element in scalar code.
- The denominator is a plain scalar (no-HVX) fp16 implementation of the same
 formula; your HVX expert must beat it by >=1.2x kernel-cycles.
