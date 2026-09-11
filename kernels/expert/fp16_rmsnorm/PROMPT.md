# fp16 RMS-norm on HVX (pure vector, no matrix engine)

Implement `candidate_kernel(const __fp16 *x, const __fp16 *gamma, __fp16 *out, int n)`
for `n == 2048` (32 HVX vectors of 64 fp16 lanes, exact tiling):

```
ms      = (1/n) * sum_i x[i]^2               (float32 accumulate)
inv_rms = 1 / sqrt(ms + 1e-3)
out[i]  = (float)x[i] * inv_rms * (float)gamma[i]
```

No mean-subtraction, no beta (this is the "no-mean-sub" RMSNorm variant).
`gamma` is a length-n per-feature runtime array (must be read, not
hardcoded). Output is compared to a float32 scalar reference with an fp16
tolerance -- HVX float arithmetic is non-IEEE (qf16), so bit-exactness is
NOT required.

- **Reduction pass:** accumulate `sum(x^2)` as a 64-lane qf16 accumulator
  vector across the n/64 blocks (`Q6_Vqf16_vmpy_VhfVhf` +
  `Q6_Vqf16_vadd_Vqf16Vqf16`). Unpack the accumulator to a scalar array ONCE
  at the end and finish the horizontal sum in float (64 scalar adds --
  negligible next to the vectorized n-element reduction). `inv_rms` is then
  a single scalar `sqrtf` call (O(1), not O(n)).
- **Scale pass:** broadcast `inv_rms` into a vector via its `__fp16` bit
  pattern and `Q6_Vh_vsplat_R`. Per block: multiply `x` by the splatted
  `inv_rms` (`Q6_Vqf16_vmpy_VhfVhf`), convert (`Q6_Vhf_equals_Vqf16`),
  multiply by the per-block loaded `gamma` vector, convert again.
- The denominator is a plain scalar (no-HVX) fp16 implementation of the
  same formula; your HVX expert must beat it by >=1.2x kernel-cycles.
