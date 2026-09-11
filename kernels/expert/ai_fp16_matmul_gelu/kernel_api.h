#ifndef KERNEL_API_H
#define KERNEL_API_H
/* FP16 32x32 matmul + GELU: out[i*N+j] = gelu( sum_k A[i*N+k]*B[k*N+j] ), N=32.
 * GELU = 0.5*x*(1+tanh(0.7978845608*(x+0.044715*x^3))), computed in float.
 * Output must be bit-exact to the harness scalar reference (float-accum -> fp16
 * round -> gelu_f32 -> fp16). HMX context is enabled before the call. */
typedef __fp16 hvx_hf;
void candidate_kernel(const hvx_hf *A, const hvx_hf *B, hvx_hf *out, int n);
#endif
