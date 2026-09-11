#ifndef KERNEL_API_H
#define KERNEL_API_H
#include <stdint.h>
/* Fused int8 linear layer: compute A*B then requantize each output to int8.
 *   acc[i*N+j] = sum_k A[i*K+k]*B[k*N+j]   (int32 intermediate)
 *   out[i*N+j] = saturate_i8( round_half_away_from_zero( (int64_t)acc * mult >> shift ) + zp )
 * A is [M x K] int8, B is [K x N] int8, out is [M x N] int8.
 * mult, shift, zp are runtime params (do not hardcode). */
void candidate_kernel(const int8_t *A, const int8_t *B, int8_t *out,
                      int M, int N, int K,
                      int32_t mult, int shift, int8_t zp);
#endif
