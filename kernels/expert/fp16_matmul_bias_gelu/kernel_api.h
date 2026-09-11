#ifndef KERNEL_API_H
#define KERNEL_API_H
typedef __fp16 hvx_hf;
/* fp16 32x32x32 matmul with a per-output-column bias add + GELU (tanh
 * approximation) FUSED into the epilogue -- pure HVX for the matmul (no
 * matrix engine); GELU has no HVX vector transcendental, so the epilogue
 * itself is necessarily scalar:
 *   acc[i][j] = sum_k A[i*k_dim+k] * B[k*n+j]     (float32-accumulate, M=N=n=K=32)
 *   m[i][j]   = (hvx_hf)acc[i][j]                  (fp16-round)
 *   t         = (float)m[i][j] + (float)bias[j]    (per-output-column bias)
 *   out[i][j] = 0.5*t*(1 + tanh(0.7978845608*(t + 0.044715*t^3)))
 * Output is compared to a float32-accumulate scalar reference with an fp16
 * tolerance -- HVX float arithmetic is non-IEEE (qf16). This is a salvage
 * op: HMX could not beat an efficient HVX qf16 baseline here (the scalar
 * GELU epilogue dominates and is paid almost identically by both matmul
 * engines), but HVX qf16's matmul speed still crushes a PLAIN SCALAR fp16
 * matmul+GELU implementation -- that scalar loop is the denominator.
 */
void candidate_kernel(const hvx_hf *A, const hvx_hf *B, const hvx_hf *bias,
                      hvx_hf *out, int n, int k_dim);
#endif
