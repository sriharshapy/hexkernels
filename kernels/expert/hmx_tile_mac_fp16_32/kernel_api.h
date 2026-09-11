#ifndef KERNEL_API_H
#define KERNEL_API_H
typedef __fp16 hvx_hf;
/* FP16 32x32 fused tile MAC on the HMX matrix engine.
 *   out[i*n+j] += sum_k A[i*n+k] * B[k*n+j]     (row-major, n=32, dtype __fp16)
 * The output buffer arrives PRE-LOADED with an accumulator tile C0; you must ADD
 * the A*B product into it (multiply-accumulate), not overwrite it. The harness
 * enables the HMX context before calling you; use VTCM scratch at HVX_VTCM_BASE
 * for crouton packing. Compared to a scalar float-accumulate reference with an
 * fp16 tolerance (HVX/HMX qf16 arithmetic is non-IEEE, so NOT bit-exact). */
void candidate_kernel(const hvx_hf *A, const hvx_hf *B, hvx_hf *out, int n);
#endif
