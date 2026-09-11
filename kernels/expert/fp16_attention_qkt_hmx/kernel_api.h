#ifndef KERNEL_API_H
#define KERNEL_API_H
typedef __fp16 hvx_hf;
/* FP16 attention QK^T scores on the HMX matrix engine, with a DEEP D=128
 * reduction (128-dim head) into a single S=32 x S=32 score tile.
 *   out[i*S+j] = sum_d Q[i*D+d] * K[j*D+d]   (row-major, S=n=32, D=k_dim=128,
 *                                              dtype __fp16; K is stored
 *                                              [S,D] -- row j IS key vector j,
 *                                              so this is Q.K^T with no
 *                                              explicit transpose of K)
 * D=128 is accumulated over four 32-deep crouton K-tiles: clear the HMX float
 * accumulator once (Q6_mxclracc_hf), then issue all four
 * (activation, weight) load-matmul pairs before the single store -- deep
 * reduction amortizes the crouton pack/unpack overhead.
 * The harness enables the HMX context before calling you; you may use VTCM
 * scratch at HVX_VTCM_BASE for crouton packing. Output is compared to a
 * scalar float-accumulate reference (cast to __fp16) with an fp16 tolerance
 * -- HVX/HMX float arithmetic is non-IEEE (qf16), so bit-exactness is NOT
 * required. */
void candidate_kernel(const hvx_hf *Q, const hvx_hf *K, hvx_hf *out, int n, int k_dim);
#endif
