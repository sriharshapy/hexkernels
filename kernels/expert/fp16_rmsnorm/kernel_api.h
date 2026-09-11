#ifndef KERNEL_API_H
#define KERNEL_API_H
typedef __fp16 hvx_hf;
/* fp16 RMS-norm over a length-n feature vector, with a per-element gamma
 * scale (no mean-subtraction, no beta -- the "no-mean-sub" RMSNorm variant):
 *   ms      = (1/n) * sum_i x[i]^2                         (float32 accumulate)
 *   inv_rms = 1 / sqrt(ms + eps)                           (eps = 1e-3f)
 *   out[i]  = (float)x[i] * inv_rms * (float)gamma[i]
 *   result cast to hvx_hf (fp16-round).
 * gamma[i]: per-feature fp16 scale (runtime, length n).
 * Output is compared to a float32 scalar reference with an fp16 tolerance --
 * HVX float arithmetic is non-IEEE (qf16). n=2048 (32 HVX vectors of 64
 * fp16 lanes) so the elementwise reduction/scale dominates over the single
 * scalar sqrt per call.
 */
void candidate_kernel(const hvx_hf *x, const hvx_hf *gamma, hvx_hf *out, int n);
#endif
