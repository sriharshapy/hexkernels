#ifndef KERNEL_API_H
#define KERNEL_API_H
typedef __fp16 hvx_hf;
/* FP16 attention A.V (scores x values) on the HMX matrix engine, with a DEEP
 * Skv=128 reduction (128 keys/values) into a single S=32 x D=32 output tile.
 *   out[i*D+j] = sum_k P[i*Skv+k] * V[k*D+j]   (row-major, S=D=n=32,
 *                                                Skv=k_dim=128, dtype __fp16;
 *                                                V is already in the generic
 *                                                B[K,N] matmul layout, no
 *                                                transpose needed)
 * Skv=128 is accumulated over four 32-deep crouton K-tiles: clear the HMX
 * float accumulator once (Q6_mxclracc_hf), then issue all four
 * (activation, weight) load-matmul pairs before the single store -- deep
 * reduction amortizes the crouton pack/unpack overhead.
 * The harness enables the HMX context before calling you; you may use VTCM
 * scratch at HVX_VTCM_BASE for crouton packing. Output is compared to a
 * scalar float-accumulate reference (cast to __fp16) with an fp16 tolerance
 * -- HVX/HMX float arithmetic is non-IEEE (qf16), so bit-exactness is NOT
 * required. */
void candidate_kernel(const hvx_hf *P, const hvx_hf *V, hvx_hf *out, int n, int k_dim);
#endif
