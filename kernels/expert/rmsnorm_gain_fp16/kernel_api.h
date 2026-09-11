#ifndef KERNEL_API_H
#define KERNEL_API_H
typedef __fp16 hvx_hf;
/*
 * Batched RMS-norm over R independent rows of length C, fp16, with a
 * PER-ROW scalar gain (NOT per-column -- this is the semantic variation
 * vs. the sibling task rmsnorm_row_fp16, which broadcasts a length-C
 * gamma[c] across all R rows; here gain[r] is a single scalar broadcast
 * uniformly across all C columns of row r).
 *
 * x/out: [R x C] hf, row-major. gain: length-R hf array (per-row scale,
 * runtime, anti-hardcode).
 *
 * Per row r, independently (compute in float, no mean-subtraction):
 *   ms        = (1/C) * sum_c ((float)x[r][c])^2      (float32 accumulate)
 *   inv_rms   = 1.0f / sqrtf(ms + 1e-3f)
 *   out[r][c] = (hvx_hf)( (float)x[r][c] * inv_rms * (float)gain[r] )
 *
 * R=6, C=80 (C NOT a multiple of 64 hf-lanes-per-HVX-vector -- tail path
 * spans one full 64-lane vector + a 16-element remainder). Output is
 * compared to a float32 scalar reference with an fp16 tolerance -- HVX
 * float arithmetic is non-IEEE (qf16), so bit-exactness is NOT required.
 */
void candidate_kernel(const hvx_hf *x, const hvx_hf *gain, hvx_hf *out, int R, int C);
#endif /* KERNEL_API_H */
