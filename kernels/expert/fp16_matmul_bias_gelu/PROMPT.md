# fp16 32x32x32 matmul + bias + GELU on HVX (pure vector matmul, scalar epilogue)

Implement `candidate_kernel(const __fp16 *A, const __fp16 *B, const __fp16 *bias,
__fp16 *out, int n, int k_dim)` for `n == k_dim == 32`:

```
acc[i][j] = sum_k A[i*k_dim+k] * B[k*n+j]                (float32-accumulate, M=N=K=32)
m[i][j]   = (__fp16)acc[i][j]                             (fp16-round)
t         = (float)m[i][j] + (float)bias[j]               (per-output-column bias)
out[i][j] = 0.5*t*(1 + tanh(0.7978845608*(t + 0.044715*t^3)))
```

- Output is compared to the float32-accumulate scalar reference with an fp16
  tolerance -- HVX float arithmetic is non-IEEE (qf16), so bit-exactness is
  NOT required.
- Vectorize the MATMUL with a row-broadcast/AXPY qf16 accumulation
  (`Q6_Vqf16_vmpy_VhfVhf` / `Q6_Vqf16_vadd_Vqf16Vqf16`) over the K=32
  reduction, then convert once (`Q6_Vhf_equals_Vqf16`).
- HVX has NO vectorized transcendental (tanh), so the bias-add + GELU
  epilogue is necessarily scalar (float conversion + bias add + cube +
  `tanhf` per output element) -- that part cannot be vectorized on this ISA.
- The denominator is a plain scalar (no-HVX) fp16 implementation of the same
  formula (matmul AND epilogue both scalar); your HVX-matmul expert must
  still beat it by >=1.2x kernel-cycles overall, because the matmul itself
  (not the epilogue) is where the vectorization win lives.
