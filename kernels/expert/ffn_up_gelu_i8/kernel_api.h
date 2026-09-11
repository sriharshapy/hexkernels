#ifndef KERNEL_API_H
#define KERNEL_API_H
#include <stdint.h>
/* Fused FFN "up" projection: GEMM + bias + saturate + GELU-via-LUT -> int8.
 *
 *   acc[i,j]  = sum_k A[i*K+k] * B[j*K+k]        (int32 accumulate)
 *   biased    = acc[i,j] + bias[j]                (int32; bias is per output column j)
 *   pre       = saturate_i8(biased)                (clamp to [-128, 127])
 *   out[i,j]  = gelu_lut[(uint8_t)(pre + 128)]      (256-entry int8 LUT, runtime)
 *
 * A is [M x K] int8 row-major (input activations).
 * B is [N x K] int8 row-major -- TRANSPOSED convention (row j of B is
 *   contiguous over K; NOT the [K x N] layout).
 * bias is [N] int32, one value per output column.
 * gelu_lut is [256] int8; index = (uint8_t)(pre + 128). It is a RUNTIME
 *   parameter -- do NOT hardcode it in the kernel. */
void candidate_kernel(const int8_t *A, const int8_t *B,
                      const int32_t *bias, const int8_t *gelu_lut,
                      int8_t *out, int M, int N, int K);
#endif /* KERNEL_API_H */
