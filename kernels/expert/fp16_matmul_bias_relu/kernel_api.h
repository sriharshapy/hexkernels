#ifndef KERNEL_API_H
#define KERNEL_API_H
typedef __fp16 hvx_hf;
/* fp16 32x32x128 (deep-K) matmul with a per-output-column bias add + ReLU
 * FUSED into the epilogue -- pure HVX (no matrix engine):
 *   acc[i][j] = sum_k A[i*k_dim+k] * B[k*n+j]     (float32-accumulate, M=N=n=32, K=k_dim=128)
 *   m[i][j]   = (hvx_hf)acc[i][j]                  (fp16-round)
 *   out[i][j] = max(0, (float)m[i][j] + (float)bias[j])
 * bias is a length-n fp16 vector (per-output-column). Output is compared to a
 * float32-accumulate scalar reference with an fp16 tolerance -- HVX float
 * arithmetic is non-IEEE (qf16), so bit-exactness is NOT required. This is a
 * salvage op: HMX could not beat an efficient HVX qf16 baseline here (thin
 * matmul-only margin erased by the fused epilogue), but HVX qf16 crushes a
 * PLAIN SCALAR fp16 implementation -- that scalar loop is the denominator.
 */
void candidate_kernel(const hvx_hf *A, const hvx_hf *B, const hvx_hf *bias,
                      hvx_hf *out, int n, int k_dim);
#endif
