#ifndef KERNEL_API_H
#define KERNEL_API_H
typedef __fp16 hvx_hf;
/*
 * Attention QKᵀ tile, fp16, COLUMN-MAJOR K variant (fp16 port of the
 * sibling int8 task qkt_tile_i8's layout).
 *
 * Q: [M x D] hf, row-major        (Q[i,d] = Q[i*D+d]).
 * K: [D x N] hf, COLUMN-major per key -- i.e. K is stored as a [D x N]
 *    matrix, K[d,j] = K[d*N+j].  (NOT [N x D] row-major.)
 * S: [M x N] hf output (scaled scores).
 *
 * Pinned formula (float32-accumulate, then two explicit hf-roundings):
 *   1. acc[i,j] = sum_d (float)Q[i*D+d] * (float)K[d*N+j]   (d = 0..D-1)
 *   2. m        = (hvx_hf)acc[i,j]                           (hf-round #1)
 *   3. S[i,j]   = (hvx_hf)((float)m * (float)scale)          (hf-round #2)
 *
 * scale: a RUNTIME hf scalar (anti-hardcode -- multiple values are swept in
 * the harness). Output is compared to a float32 scalar reference with an
 * fp16 tolerance -- HVX float arithmetic is non-IEEE (qf16), so
 * bit-exactness is NOT required.
 *
 * M=8, N=8, D=40 (D=40 < 64 hf-lanes-per-HVX-vector -- the whole reduction
 * fits in one vector once zero-padded; exercises the zero-pad-tail path).
 *
 * NOTE: `scale` is typed `_Float16` (not `hvx_hf`/`__fp16`) purely because
 * this hexagon-clang target rejects `__fp16` BY VALUE in a function
 * parameter ("parameters cannot have __fp16 type") -- there is no ABI
 * lowering for a scalar fp16 argument. `_Float16` is bit-for-bit the same
 * IEEE-754 binary16 format and IS accepted by value; it is still a genuine
 * runtime fp16 scalar (not hardcoded, not silently widened to float).
 */
void candidate_kernel(const hvx_hf *Q, const hvx_hf *K, hvx_hf *S,
                      int M, int N, int D, _Float16 scale);
#endif /* KERNEL_API_H */
