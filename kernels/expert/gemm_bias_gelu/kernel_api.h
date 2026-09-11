#ifndef KERNEL_API_H
#define KERNEL_API_H
#include <stdint.h>
/* Fused int8 GEMM + bias + GELU-via-LUT -> int8:
 *   acc[i*N+j]  = sum_k A[i*K+k] * B[k*N+j]   (int32 intermediate)
 *   biased       = acc[i*N+j] + bias[j]         (int32)
 *   pre_lut      = saturate_i8(biased)          (clamp to [-128,127])
 *   out[i*N+j]  = gelu_lut[(uint8_t)(pre_lut + 128)]  (256-entry int8 LUT, index = val+128)
 *
 * A is [M x K] int8 row-major, B is [K x N] int8 row-major.
 * bias is [N] int32 (per output column).
 * gelu_lut is [256] int8; index = (uint8_t)(pre_lut + 128).
 * Do NOT hardcode the LUT -- it is a runtime parameter. */
void candidate_kernel(const int8_t *A, const int8_t *B,
                      const int32_t *bias,
                      const int8_t *gelu_lut,
                      int8_t *out,
                      int M, int N, int K);
#endif
