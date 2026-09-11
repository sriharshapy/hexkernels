#ifndef KERNEL_API_H
#define KERNEL_API_H
typedef __fp16 hvx_hf;
/* fp16 32x32x32 matmul + elementwise residual add (skip connection) FUSED
 * into the epilogue -- pure HVX (no matrix engine):
 *   acc[i][j] = sum_k A[i*k_dim+k] * B[k*n+j]     (float32-accumulate, M=N=K=32)
 *   m[i][j]   = (hvx_hf)acc[i][j]                  (fp16-round)
 *   out[i][j] = (float)m[i][j] + (float)C[i*n+j]   (elementwise residual, same n x n shape)
 * Output is compared to a float32-accumulate scalar reference with an fp16
 * tolerance -- HVX float arithmetic is non-IEEE (qf16). This is a salvage
 * op: HMX could not beat an efficient HVX qf16 baseline here (reading the
 * second full n x n operand C costs real memory cycles paid almost
 * identically by both matmul engines, swamping the thin matmul-only
 * margin), but HVX qf16 still crushes a PLAIN SCALAR fp16 matmul+residual
 * implementation -- that scalar loop is the denominator.
 */
void candidate_kernel(const hvx_hf *A, const hvx_hf *B, const hvx_hf *C,
                      hvx_hf *out, int n, int k_dim);
#endif
