# Batched fp16 RMS-norm on HVX (pure vector, no matrix engine)

Implement `candidate_kernel(const __fp16 *x, const __fp16 *gamma, __fp16 *out, int R, int C)`
for `R == 5, C == 90`. `x`/`out` are `[R x C]` row-major; `gamma` is a length-C
array SHARED (broadcast) across all R rows. Each row is normalized
INDEPENDENTLY by its own RMS:

```
ms        = (1/C) * sum_c ((float)x[r][c])^2
inv_rms   = 1 / sqrt(ms + 1e-3)
out[r][c] = (float)x[r][c] * inv_rms * (float)gamma[c]
```

No mean-subtraction, no beta (the "no-mean-sub" RMSNorm variant, same as
fp16_rmsnorm but batched over R rows with gamma broadcast per-row instead of
applied to a single vector). `gamma` must be read at runtime (not
hardcoded). Output is compared to a float32 scalar reference with an fp16
tolerance -- HVX float arithmetic is non-IEEE (qf16), so bit-exactness is
NOT required.

C=90 is NOT a multiple of 64 fp16-lanes-per-HVX-vector -- handle the tail
path (one full 64-lane vector + 26-element remainder per row). Edge cases
include an all-zero row (ms=0, relies on the +1e-3 epsilon to avoid a
divide-by-zero), a row with one dominant large-magnitude element, and
gamma values that are both negative and positive.

Implement ONLY this function (Hexagon HVX C). Include <hexagon_types.h> and <hexagon_protos.h>. Do NOT write main(). Respond with a single complete C code block.
