# Batched fp16 RMS-norm with a PER-ROW gain on HVX (pure vector, no matrix engine)

Implement `candidate_kernel(const __fp16 *x, const __fp16 *gain, __fp16 *out, int R, int C)`
for `R == 6, C == 80`. `x`/`out` are `[R x C]` row-major; `gain` is a length-R
array -- ONE scalar per row, broadcast uniformly across all C columns of
that row (NOT a per-column gamma[C] -- do not confuse this with the sibling
task rmsnorm_row_fp16). Each row is normalized INDEPENDENTLY by its own RMS:

```
ms        = (1/C) * sum_c ((float)x[r][c])^2
inv_rms   = 1 / sqrt(ms + 1e-3)
out[r][c] = (float)x[r][c] * inv_rms * (float)gain[r]
```

No mean-subtraction, no beta. `gain` must be read at runtime (not
hardcoded, and it is indexed by ROW r, never by column c). Output is
compared to a float32 scalar reference with an fp16 tolerance -- HVX float
arithmetic is non-IEEE (qf16), so bit-exactness is NOT required.

C=80 is NOT a multiple of 64 fp16-lanes-per-HVX-vector -- handle the tail
path (one full 64-lane vector + a 16-element remainder per row). Edge cases
include a near-zero row (ms relies on the +1e-3 epsilon to avoid
divide-by-zero/inf), a row with one dominant large-magnitude element, and
gain values that are both negative and positive.

Implement ONLY this function (Hexagon HVX C). Include <hexagon_types.h> and <hexagon_protos.h>. Do NOT write main(). Respond with a single complete C code block.
