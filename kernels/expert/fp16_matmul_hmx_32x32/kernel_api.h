#ifndef KERNEL_API_H
#define KERNEL_API_H
/* FP16 32x32 matmul on the HMX matrix engine.
 *   out[i*N+j] = sum_k A[i*N+k] * B[k*N+j]      (row-major, N=32, dtype __fp16)
 * The harness enables the HMX context before calling you; you may use VTCM
 * scratch at HVX_VTCM_BASE for crouton packing. Output is compared to a scalar
 * float-accumulate reference (cast to __fp16) with an fp16 tolerance -- HVX/HMX
 * float arithmetic is non-IEEE (qf16), so bit-exactness is NOT required. */
typedef __fp16 hvx_hf;
void candidate_kernel(const hvx_hf *A, const hvx_hf *B, hvx_hf *out, int n);
#endif
