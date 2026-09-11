#ifndef KERNEL_API_H
#define KERNEL_API_H
#include <stdint.h>
/* Int8 matmul + requantize + residual (skip-connection) add -> int8.
 *
 * A is [M x K] int8 row-major, B is [K x N] int8 row-major CONVENTIONAL
 * layout (B[k,j] = B[k*N+j]). residual is [M x N] int8 (the skip-connection
 * to add). out is [M x N] int8.
 *
 *   acc[i,j] = sum_k A[i*K+k] * B[k*N+j]              (int32 accumulate)
 *   r    = (int64_t)acc[i,j] * (int64_t)scale_mult
 *   half = (scale_shift > 0) ? (1LL << (scale_shift-1)) : 0
 *   q    = (r >= 0) ? ((r + half) >> scale_shift) : -((-r + half) >> scale_shift)
 *   requantized = clamp(q, -128, 127)
 *   out[i*N+j] = clamp((int32_t)requantized + (int32_t)residual[i*N+j], -128, 127)
 *
 * NOTE: the residual is added AFTER the requantize/shift step (a second,
 * separate saturating add) -- NOT before. scale_mult, scale_shift are
 * RUNTIME parameters (anti-hardcode; multiple pairs are swept).
 * M=8, N=8, K=90 (90 = 22*4 + 2 is NOT a multiple of 4 -- reduction tail).
 */
void candidate_kernel(const int8_t *A, const int8_t *B, const int8_t *residual,
                      int8_t *out, int M, int N, int K,
                      int32_t scale_mult, int scale_shift);
#endif /* KERNEL_API_H */
