# fp16 layer-norm on HVX (pure vector, no matrix engine)

Implement `candidate_kernel(const __fp16 *x, const __fp16 *gamma, const __fp16 *beta,
__fp16 *out, int n)` for `n == 2048` (32 HVX vectors of 64 fp16 lanes, exact tiling):

```
mean    = (1/n) * sum_i x[i]                          (float32 accumulate)
var     = (1/n) * sum_i x[i]^2  -  mean^2               (float32 accumulate)
inv_std = 1 / sqrt(var + 1e-3)
out[i]  = ((float)x[i] - mean) * inv_std * (float)gamma[i] + (float)beta[i]
```

`gamma`/`beta` are length-n per-feature runtime arrays (must be read, not
hardcoded). Output is compared to a float32 scalar reference with an fp16
tolerance -- HVX float arithmetic is non-IEEE (qf16), so bit-exactness is
NOT required.

- **Reduction pass:** accumulate `sum(x)` and `sum(x^2)` as 64-lane qf16
  accumulator vectors across the n/64 blocks (`Q6_Vqf16_vadd_Vqf16Vhf` for
  the sum, `Q6_Vqf16_vmpy_VhfVhf` + `Q6_Vqf16_vadd_Vqf16Vqf16` for the sum of
  squares). Unpack BOTH 64-lane accumulators to scalar arrays once at the
  end and finish the horizontal sum in float (64 scalar adds -- negligible
  next to the vectorized n-element reduction). `mean`/`var`/`inv_std` are
  then plain scalars computed with ONE scalar `sqrtf` call (O(1), not O(n)).
- **Affine pass:** broadcast `mean` and `inv_std` into vectors by taking
  their `__fp16` bit pattern and using `Q6_Vh_vsplat_R` (the same
  bit-pattern-splat idiom used for scalar row-broadcast in matmul kernels).
  Per block: `Q6_Vqf16_vsub_VhfVhf(x, meanVec)` then convert
  (`Q6_Vhf_equals_Vqf16`), multiply by the splatted `inv_std`, multiply by
  the per-block loaded `gamma` vector, add the per-block loaded `beta`
  vector -- all in qf16 with a convert-to-hf after each op (v68 lacks a
  native `Vhf+Vhf`/`Vhf*Vhf` op; the qf16 path is what's native).
- The denominator is a plain scalar (no-HVX) fp16 implementation of the
  same formula; your HVX expert must beat it by >=1.2x kernel-cycles.
