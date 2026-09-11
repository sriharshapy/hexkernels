#ifndef KERNEL_API_H
#define KERNEL_API_H
typedef __fp16 hvx_hf;
/* FP16 matmul with a DEEP K reduction on the HMX matrix engine.
 *   out[i*n+j] = sum_k A[i*k_dim+k] * B[k*n+j]   (row-major, M=N=n=32, K=k_dim=128)
 * Accumulate K over four 32-deep crouton K-tiles: clear the HMX float accumulator
 * once (Q6_mxclracc_hf), issue all four (activation,weight) load-matmul pairs,
 * then a single store. The harness enables the HMX context before calling you;
 * use VTCM scratch at HVX_VTCM_BASE. Compared with an fp16 tolerance (non-IEEE). */
void candidate_kernel(const hvx_hf *A, const hvx_hf *B, hvx_hf *out, int n, int k_dim);
#endif
