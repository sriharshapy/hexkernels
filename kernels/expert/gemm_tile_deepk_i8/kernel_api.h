#ifndef KERNEL_API_H
#define KERNEL_API_H
#include <stdint.h>
/* Deep-K int8 GEMM tile, CONVENTIONAL (non-transposed) B layout:
 *   C[i*N+j] = sum_k A[i*K+k] * B[k*N+j]     (int32 accumulate, k = 0..K-1)
 *
 * A is [M x K] int8 row-major (A[i,k] = A[i*K+k]).
 * B is [K x N] int8 row-major, the CONVENTIONAL GEMM operand layout -- row k
 * of B is contiguous over j (B[k,j] = B[k*N+j]). This is NOT transposed
 * (contrast with the sibling task gemm_tile_i8, whose B is [N x K]).
 * C is [M x N] int32 output, no requantization.
 *
 * M=8, N=8, K=262 (deep-K; 262 = 65*4 + 2 is NOT a multiple of 4, so a
 * vrmpy-style 4-byte-group reduction has a genuine 2-byte tail group after
 * 65 full groups). max |acc| = 127*127*262 ~= 4.22e6, well within int32.
 */
void candidate_kernel(const int8_t *A, const int8_t *B, int32_t *C,
                      int M, int N, int K);
#endif /* KERNEL_API_H */
