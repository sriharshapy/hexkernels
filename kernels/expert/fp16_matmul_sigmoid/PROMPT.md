# fp16 32x32x32 matmul + sigmoid on HVX (pure vector matmul, scalar epilogue)

Implement `candidate_kernel(const __fp16 *A, const __fp16 *B, __fp16 *out, int n, int k_dim)`
for `n == k_dim == 32`:

```
acc[i][j] = sum_k A[i*k_dim+k] * B[k*n+j]   (float32-accumulate, M=N=K=32)
m[i][j]   = (__fp16)acc[i][j]                (fp16-round)
out[i][j] = 1 / (1 + exp(-(float)m[i][j]))
```

- Output is compared to the float32-accumulate scalar reference with an fp16
  tolerance -- HVX float arithmetic is non-IEEE (qf16), so bit-exactness is
  NOT required.
- Vectorize the MATMUL with a row-broadcast/AXPY qf16 accumulation
  (`Q6_Vqf16_vmpy_VhfVhf` / `Q6_Vqf16_vadd_Vqf16Vqf16`) over the K=32
  reduction, then convert once (`Q6_Vhf_equals_Vqf16`).
- HVX has NO vectorized transcendental (exp), so the sigmoid epilogue is
  necessarily scalar (float conversion + `expf` per output element).
- The denominator is a plain scalar (no-HVX) fp16 implementation of the same
  formula; your HVX-matmul expert must still beat it by >=1.2x kernel-cycles
  overall -- the vectorization win comes from the matmul, not the epilogue.
