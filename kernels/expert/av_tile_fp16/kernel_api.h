#ifndef KERNEL_API_H
#define KERNEL_API_H
typedef __fp16 hvx_hf;
/*
 * Attention A·V tile, fp16, COLUMN-MAJOR V variant (fp16 port of the
 * sibling int8 task av_tile_i8's layout, no requantization).
 *
 * A: [M x N] hf attention probs, row-major (A[i,j] = A[i*N+j]).
 * V: [D x N] hf value matrix, COLUMN-major/pre-transposed -- i.e. V is
 *    stored as a [D x N] matrix, V[d,j] = V[d*N+j]. (NOT [N x D] row-major.
 *    Both A and V are row-major-contiguous over the REDUCTION axis j, a
 *    clean dot-product-per-output-element shape.)
 * O: [M x D] hf output, row-major.
 *
 * Pinned formula (float32-accumulate, single hf-rounding):
 *   O[i,d] = (hvx_hf)( sum_j (float)A[i*N+j] * (float)V[d*N+j] )   (j=0..N-1)
 *
 * M=8, D=8, N=40 (N is the reduction axis, < 64 hf-lanes-per-HVX-vector --
 * zero-pad-tail path). Output is compared to a float32 scalar reference
 * with an fp16 tolerance -- HVX float arithmetic is non-IEEE (qf16), so
 * bit-exactness is NOT required.
 */
void candidate_kernel(const hvx_hf *A, const hvx_hf *V, hvx_hf *O,
                      int M, int N, int D);
#endif /* KERNEL_API_H */
