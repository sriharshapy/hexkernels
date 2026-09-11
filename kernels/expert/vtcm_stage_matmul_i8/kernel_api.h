#ifndef KERNEL_API_H
#define KERNEL_API_H
#include <stdint.h>
/*
 * GEMM with a LARGE streamed operand A and a SMALL staged operand X:
 *   C[m*N+j] = sum_{k=0}^{K-1} A[m*K+k] * X[k*N+j]   for m in [0,M), j in [0,N)
 *
 * A is [M x K] row-major (row m contiguous over k) -- the LARGE operand,
 * streamed one row at a time. X is [K x N] row-major (row k contiguous over
 * j) -- the SMALL operand (N<=4), identical/reused for EVERY one of the M
 * rows, so it should be packed/staged ONCE, not per row.
 *
 * int32 accumulation is safe at these bounds: |A[i]|,|X[i]| <= 127, so
 * |sum| <= 127*127*K, comfortably inside int32 range for the given K.
 *
 * Fixed sizes: M=161, K=8101 (NOT a multiple of 4 -- vrmpy k-group tail),
 * N=4. A is ~1.24MB (exceeds L2 -> DDR-bandwidth-bound on the A side); X is
 * only ~32KB and constant across all rows.
 */
void candidate_kernel(const int8_t *A, const int8_t *X, int32_t *C, int M, int K, int N);
#endif
