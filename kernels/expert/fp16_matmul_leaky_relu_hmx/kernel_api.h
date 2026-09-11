#ifndef KERNEL_API_H
#define KERNEL_API_H
typedef __fp16 hvx_hf;
/* HMX fp16 32x32x128 (deep-K) matmul with leaky ReLU (slope 1/8) FUSED into
 * the epilogue:
 *   acc[i][j] = sum_k A[i*k_dim+k] * B[k*n+j]   (float32-accumulate, M=N=n=32, K=k_dim=128)
 *   m[i][j]   = (hvx_hf)acc[i][j]                (fp16-round: matches HMX output)
 *   v         = (float)m[i][j]
 *   out[i][j] = v > 0 ? v : v * 0.125f
 * The harness enables the HMX context before calling you; VTCM scratch at
 * HVX_VTCM_BASE for crouton packing. Output is compared to a float32-accumulate
 * scalar reference with an fp16 tolerance -- HVX/HMX float arithmetic is
 * non-IEEE (qf16).
 */
void candidate_kernel(const hvx_hf *A, const hvx_hf *B, hvx_hf *out, int n, int k_dim);
#endif
