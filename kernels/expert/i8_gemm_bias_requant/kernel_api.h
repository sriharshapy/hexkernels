#ifndef KERNEL_API_H
#define KERNEL_API_H
#include <stdint.h>
/* Fused int8 GEMM + bias + requantize (NO relu):
 *   acc[i*N+j]  = sum_k A[i*K+k] * B[k*N+j]   (int32 intermediate)
 *   biased       = acc[i*N+j] + bias[j]         (int32)
 *   v            = (int64_t)biased * mult
 *   half         = shift > 0 ? (1LL << (shift-1)) : 0
 *   r            = (v >= 0) ? (v+half)>>shift : -(((-v)+half)>>shift)  // round half away from 0
 *   out[i*N+j]  = saturate_i8(r + zp)          (clamp to [-128, 127])
 *
 * A is [M x K] int8 row-major, B is [K x N] int8 row-major.
 * bias is [N] int32 (one per output column).
 * mult, shift, zp are scalar runtime params -- do NOT hardcode them. */
void candidate_kernel(const int8_t *A, const int8_t *B,
                      const int32_t *bias, int8_t *out,
                      int M, int N, int K,
                      int32_t mult, int shift, int8_t zp);
#endif
