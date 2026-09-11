#ifndef KERNEL_API_H
#define KERNEL_API_H
typedef __fp16 hvx_hf;
/* fp16 32x32x32 matmul with sigmoid FUSED into the epilogue -- pure HVX for
 * the matmul (no matrix engine); sigmoid has no HVX vector transcendental,
 * so the epilogue itself is necessarily scalar:
 *   acc[i][j] = sum_k A[i*k_dim+k] * B[k*n+j]   (float32-accumulate, M=N=K=32)
 *   m[i][j]   = (hvx_hf)acc[i][j]                (fp16-round)
 *   out[i][j] = 1 / (1 + exp(-(float)m[i][j]))
 * Output is compared to a float32-accumulate scalar reference with an fp16
 * tolerance -- HVX float arithmetic is non-IEEE (qf16). Salvage op: HMX
 * could not beat an efficient HVX qf16 baseline here (the scalar sigmoid
 * epilogue dominates and is paid almost identically by both matmul
 * engines), but HVX qf16's matmul speed still crushes a PLAIN SCALAR fp16
 * matmul+sigmoid implementation -- that scalar loop is the denominator.
 */
void candidate_kernel(const hvx_hf *A, const hvx_hf *B, hvx_hf *out, int n, int k_dim);
#endif
