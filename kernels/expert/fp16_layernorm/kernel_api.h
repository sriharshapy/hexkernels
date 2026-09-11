#ifndef KERNEL_API_H
#define KERNEL_API_H
typedef __fp16 hvx_hf;
/* fp16 layer-norm over a length-n feature vector, with a per-element affine
 * (gamma, beta), computed with a float32-accumulate mean/variance:
 *   mean    = (1/n) * sum_i x[i]                          (float32 accumulate)
 *   var     = (1/n) * sum_i x[i]^2  -  mean^2               (float32 accumulate)
 *   inv_std = 1 / sqrt(var + eps)                          (eps = 1e-3f)
 *   out[i]  = (float)(((float)x[i]-mean) * inv_std) * (float)gamma[i] + (float)beta[i]
 *   result cast to hvx_hf (fp16-round).
 * gamma[i]/beta[i]: per-feature fp16 affine params (runtime, length n).
 * Output is compared to a float32 scalar reference with an fp16 tolerance --
 * HVX float arithmetic is non-IEEE (qf16). n=2048 (32 HVX vectors of 64
 * fp16 lanes) so the elementwise reduction/affine transform dominates over
 * the single scalar sqrt per call.
 */
void candidate_kernel(const hvx_hf *x, const hvx_hf *gamma, const hvx_hf *beta,
                      hvx_hf *out, int n);
#endif
