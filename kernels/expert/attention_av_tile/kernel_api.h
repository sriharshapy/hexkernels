#ifndef KERNEL_API_H
#define KERNEL_API_H
#include <stdint.h>
/*
 * Attention A·V tile: compute output O = A · V.
 *
 * A: [M x N] int8 attention scores (row-major).
 * V: [N x D] int8 value matrix    (row-major).
 * O: [M x D] int32 output         (row-major, accumulate without requant).
 *
 * Pinned formula:
 *   O[i,d] = sum_j  A[i*N+j] * V[j*D+d]     (int32 accumulate)
 *
 * M=16, N=16, D=130 (D is NOT a multiple of 128 — handle the reduction tail).
 *
 * Output is int32 (no requantization in this tile — requant is a separate pass).
 * This lets the candidate focus on the GEMM structure with the non-128 D tail.
 */
void candidate_kernel(const int8_t *A, const int8_t *V, int32_t *O,
                      int M, int N, int D);
#endif /* KERNEL_API_H */
