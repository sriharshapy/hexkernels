#ifndef KERNEL_API_H
#define KERNEL_API_H
typedef __fp16 hvx_hf;
/*
 * Batched RMS-norm over R independent rows of length C, fp16. x/out are
 * [R x C] row-major; gamma is length-C and SHARED (broadcast) across all R
 * rows (this is the layout variation on top of fp16_rmsnorm's single-vector
 * original -- same no-mean-sub RMSNorm formula, per-feature gain, now
 * broadcast per-row instead of applied to a single vector).
 *
 * Per row r, independently (compute in float, matching the fp16-boundary
 * style of the original):
 *   ms        = (1/C) * sum_c ((float)x[r][c])^2            (float32 accumulate)
 *   inv_rms   = 1.0f / sqrtf(ms + 1e-3f)
 *   out[r][c] = (hvx_hf)( (float)x[r][c] * inv_rms * (float)gamma[c] )
 *
 * gamma[c]: per-feature fp16 scale (runtime, length C, shared across rows).
 * Output is compared to a float32 scalar reference with an fp16 tolerance --
 * HVX float arithmetic is non-IEEE (qf16). R=5, C=90 (C NOT a multiple of 64
 * fp16-lanes-per-HVX-vector -- real tail path).
 */
void candidate_kernel(const hvx_hf *x, const hvx_hf *gamma, hvx_hf *out, int R, int C);
#endif
