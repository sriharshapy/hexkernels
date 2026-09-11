#ifndef KERNEL_API_H
#define KERNEL_API_H
#include <stdint.h>
/* Row-major int8 matmul, B given TRANSPOSED: C[i*N+j] = sum_k A[i*K+k]*B[j*K+k],
 * int32 accumulate (no saturation needed -- int32 is wide enough at these sizes:
 * max |acc| = 128*128*K <= 1.6e6). A is [M x K] row-major (row i contiguous over
 * k). B is [N x K] row-major (row j contiguous over k) -- i.e. B is the TRANSPOSE
 * of the conventional [K x N] GEMM operand, so both A and B are contiguous over
 * the reduction axis K (a dot-product-shaped access pattern, distinct from the
 * conventional-layout i8_gemm_tile task). */
void candidate_kernel(const int8_t *A, const int8_t *B, int32_t *C, int M, int N, int K);
#endif
