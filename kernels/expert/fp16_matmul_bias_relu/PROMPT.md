# fp16 32x32x128 (deep-K) matmul + bias + ReLU on HVX (pure vector, no matrix engine)

Implement `candidate_kernel(const __fp16 *A, const __fp16 *B, const __fp16 *bias,
__fp16 *out, int n, int k_dim)` for `n == 32`, `k_dim == 128`:

```
acc[i][j] = sum_k A[i*k_dim+k] * B[k*n+j]            (float32-accumulate, M=N=32, K=128)
m[i][j]   = (__fp16)acc[i][j]                         (fp16-round)
out[i][j] = max(0, (float)m[i][j] + (float)bias[j])   (per-output-column bias, then ReLU)
```

- Output is compared to the float32-accumulate scalar reference with an fp16
  tolerance -- HVX float arithmetic is non-IEEE (qf16), so bit-exactness is
  NOT required.
- Use a row-broadcast/AXPY qf16 matmul: for each output row i, splat each
  `A[i][k]` scalar across a vector, multiply against the vector-loaded row
  `B[k][:]`, and accumulate in qf16 (`Q6_Vqf16_vmpy_VhfVhf` / `Q6_Vqf16_vadd_Vqf16Vqf16`)
  over the full K=128 reduction.
- Fuse the bias-add + ReLU epilogue as native HVX vector ops directly on the
  matmul result vector -- do NOT cast to float and process element-by-element
  (that round-trips through the software hf<->float conversion routine twice
  per element and swamps the whole kernel).
- v68 has no native `Vhf+Vhf` add (needs v79+): do the bias add in qf16
  (`Q6_Vqf16_vadd_VhfVhf`, native at v68), convert once
  (`Q6_Vhf_equals_Vqf16`), then ReLU with the native `Q6_Vhf_vmax_VhfVhf`.
- The denominator is a plain scalar (no-HVX) fp16 implementation of the same
  formula; your HVX expert must beat it by >=1.2x kernel-cycles.
