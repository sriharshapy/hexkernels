#ifndef KERNEL_API_H
#define KERNEL_API_H
typedef __fp16 hvx_hf;
/* fp16 32x32 matmul on the HMX matrix engine with a GELU (tanh approximation)
 * epilogue FUSED after the crouton unpack (no bias term):
 *   acc[i][j] = sum_k A[i*n+k] * B[k*n+j]     (float32-accumulate, n=32)
 *   m[i][j]   = (hvx_hf)acc[i][j]              (fp16-round)
 *   out[i][j] = 0.5*m*(1 + tanh(0.7978845608*(m + 0.044715*m^3)))
 * The harness enables the HMX context before calling you; use VTCM scratch at
 * HVX_VTCM_BASE for crouton packing (layout off(r,c)=(r/2)*64+c*2+(r&1)).
 * GELU has no HVX vector transcendental, so the epilogue itself is necessarily
 * scalar -- but the matmul is done on the matrix engine. Output is compared to
 * a float32-accumulate scalar reference with an fp16 tolerance (HVX/HMX qf16
 * arithmetic is non-IEEE, so bit-exactness is NOT required).
 */
void candidate_kernel(const hvx_hf *A, const hvx_hf *B, hvx_hf *out, int n);
#endif
