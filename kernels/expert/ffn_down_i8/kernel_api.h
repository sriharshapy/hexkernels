#ifndef KERNEL_API_H
#define KERNEL_API_H
#include <stdint.h>
/* Fused FFN "down" projection: GEMM + bias + int64 round-half-away-from-zero
 * requantize -> int8. NO activation function (this is the final FFN layer,
 * just bias + requantize).
 *
 *   acc[i,j]  = sum_k A[i*K+k] * B[j*K+k]         (int32 accumulate)
 *   biased    = acc[i,j] + bias[j]                 (int32; bias is per output column j)
 *   r         = (int64_t)biased * (int64_t)scale_mult
 *   half      = (scale_shift > 0) ? (1LL << (scale_shift-1)) : 0
 *   q         = (r >= 0) ? ((r + half) >> scale_shift) : -((-r + half) >> scale_shift)
 *   out[i,j]  = clamp(q, -128, 127)
 *
 * A is [M x K] int8 row-major (the FFN hidden/intermediate activations, K = Dff).
 * B is [N x K] int8 row-major -- TRANSPOSED convention (row j of B is
 *   contiguous over K; NOT the [K x N] layout). This is the down-projection weight.
 * bias is [N] int32, one value per output column.
 * scale_mult (int32) / scale_shift (int) are RUNTIME parameters (swept by the
 *   harness) -- do NOT hardcode them. */
void candidate_kernel(const int8_t *A, const int8_t *B, const int32_t *bias,
                      int8_t *out, int M, int N, int K,
                      int32_t scale_mult, int scale_shift);
#endif /* KERNEL_API_H */
