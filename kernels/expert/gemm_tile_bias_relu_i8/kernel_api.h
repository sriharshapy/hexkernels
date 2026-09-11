#ifndef KERNEL_API_H
#define KERNEL_API_H
#include <stdint.h>
/* Fused int8 GEMM tile + per-column bias + ReLU, output narrowed to int8.
 * A is [M x K] row-major (row i contiguous over k). B is [N x K] row-major
 * (TRANSPOSED relative to the conventional [K x N] GEMM operand -- row j
 * contiguous over k, same convention as gemm_tile_i8). `bias` is int32[N],
 * ONE value per output COLUMN j, broadcast across all rows i.
 *
 *   acc   = sum_k A[i*K+k]*B[j*K+k] + bias[j]     (bias added BEFORE relu)
 *   relu  = acc > 0 ? acc : 0
 *   C[i*N+j] = (int8_t) clamp(relu, 0, 127)        (relu already >=0; the lower
 *                                                    clamp is kept for defensive
 *                                                    clarity, the upper clamp is
 *                                                    the one that actually bites)
 */
void candidate_kernel(const int8_t *A, const int8_t *B, const int32_t *bias, int8_t *C,
                      int M, int N, int K);
#endif
