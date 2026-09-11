#ifndef KERNEL_API_H
#define KERNEL_API_H
typedef __fp16 hvx_hf;
/* FP16 1x1 convolution (= matmul over channels) on the HMX matrix engine.
 *   out[p*n+co] = sum_ci In[p*n+ci] * W[co*n+ci]   (n=32: P pixels, Cin=Cout=n)
 * In is [P x Cin] row-major activations, W is [Cout x Cin] row-major weights
 * (weight-stationary NCHW convention); a 1x1 conv contracts the channel axis, so
 * out = In * W^T. The harness enables the HMX context before calling you; use
 * VTCM scratch at HVX_VTCM_BASE. Compared with an fp16 tolerance (non-IEEE). */
void candidate_kernel(const hvx_hf *In, const hvx_hf *W, hvx_hf *out, int n);
#endif
