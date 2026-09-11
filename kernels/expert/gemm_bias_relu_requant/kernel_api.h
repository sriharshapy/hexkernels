#ifndef KERNEL_API_H
#define KERNEL_API_H
#include <stdint.h>
/* Fused int8 GEMM + bias + ReLU + requantize:
 *   acc[i*N+j]  = sum_k A[i*K+k] * B[k*N+j]   (int32 intermediate, i8*i8 products)
 *   biased       = acc[i*N+j] + bias[j]         (int32)
 *   after_relu   = max(biased, 0)               (ReLU -- clamp to 0 if negative)
 *   out[i*N+j]  = saturate_i8(
 *                     round_half_away_from_zero( (int64_t)after_relu * mult >> shift ) + zp )
 *
 * A is [M x K] int8 row-major, B is [K x N] int8 row-major.
 * bias is [N] int32 (one bias per output column, i.e. per output channel).
 * mult, shift, zp are scalar runtime params -- do NOT hardcode them. */
void candidate_kernel(const int8_t *A, const int8_t *B,
                      const int32_t *bias, int8_t *out,
                      int M, int N, int K,
                      int32_t mult, int shift, int8_t zp);
#endif
