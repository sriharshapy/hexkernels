#ifndef KERNEL_API_H
#define KERNEL_API_H
typedef __fp16 hvx_hf;
/* FP16 32x32 matmul + ReLU on the HMX matrix engine.
 *   out[i*n+j] = max(sum_k A[i*n+k] * B[k*n+j], 0)   (row-major, n=32, __fp16)
 * The harness enables the HMX context before calling you; use VTCM scratch at
 * HVX_VTCM_BASE for crouton packing. Compared with an fp16 tolerance (HVX/HMX
 * qf16 arithmetic is non-IEEE). */
void candidate_kernel(const hvx_hf *A, const hvx_hf *B, hvx_hf *out, int n);
#endif
