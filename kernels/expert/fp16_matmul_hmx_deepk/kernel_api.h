#ifndef KERNEL_API_H
#define KERNEL_API_H
typedef __fp16 hvx_hf;
/* FP16 32x32 matmul with a DEEP K=128 reduction on the HMX matrix engine.
 *   out[i*n+j] = sum_k A[i*k_dim+k] * B[k*n+j]   (row-major, M=N=n=32, K=k_dim=128,
 *                                                 dtype __fp16)
 * K=128 is accumulated over four 32-deep crouton K-tiles: clear the HMX float
 * accumulator once (Q6_mxclracc_hf), then issue all four (activation, weight)
 * load-matmul pairs before the single store -- deep reduction amortizes the
 * crouton pack/unpack overhead over 4x the compute of the single-tile
 * fp16_matmul_hmx_32x32 sibling.
 * The harness enables the HMX context before calling you; you may use VTCM
 * scratch at HVX_VTCM_BASE for crouton packing. Output is compared to a scalar
 * float-accumulate reference (cast to __fp16) with an fp16 tolerance -- HVX/HMX
 * float arithmetic is non-IEEE (qf16), so bit-exactness is NOT required.
 */
void candidate_kernel(const hvx_hf *A, const hvx_hf *B, hvx_hf *out, int n, int k_dim);
#endif
