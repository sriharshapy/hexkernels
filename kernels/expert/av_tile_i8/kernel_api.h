#ifndef KERNEL_API_H
#define KERNEL_API_H
#include <stdint.h>
/*
 * Attention A·V tile, COLUMN-MAJOR (pre-transposed) V variant: compute
 * output O = A · V, no requantization.
 *
 * A: [M x N] int8 attention scores (row-major, A[i,j] = A[i*N+j]).
 * V: [D x N] int8 value matrix, COLUMN-major/pre-transposed -- i.e. V is
 *    stored as a [D x N] matrix, V[d,j] = V[d*N+j]. (NOT [N x D] row-major
 *    -- this is the layout variation vs. the sibling task
 *    attention_av_tile. Both A and V are now row-major-contiguous over the
 *    REDUCTION axis j, giving a clean dot-product-per-output-element
 *    shape.)
 * O: [M x D] int32 output (row-major, accumulate without requant).
 *
 * Pinned formula:
 *   O[i,d] = sum_j  A[i*N+j] * V[d*N+j]     (int32 accumulate, j=0..N-1)
 *
 * M=16, N=16, D=90 (D is NOT a multiple of 32 -- the output loop has a tail
 * when processed in 32-wide blocks).
 *
 * Output is int32 (no requantization in this tile).
 */
void candidate_kernel(const int8_t *A, const int8_t *V, int32_t *O,
                      int M, int N, int D);
#endif /* KERNEL_API_H */
