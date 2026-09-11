#ifndef KERNEL_API_H
#define KERNEL_API_H
#include <stdint.h>
/*
 * Attention QKᵀ tile: compute scaled dot-product scores S = Q · Kᵀ, requantized.
 *
 * Q: [M x D] int8, row-major.
 * K: [N x D] int8, row-major.   (K is already transposed: K[j,d] = K_orig[j,d])
 * S: [M x N] int8 output (requantized scores).
 *
 * Pinned multi-step integer formula:
 *   1. raw[i,j] = sum_d Q[i*D+d] * K[j*D+d]          (int32 accumulate, no overflow for
 *                                                       D<=130, int8 inputs: max |val|= D*127*127)
 *   2. Requantize raw[i,j] -> int8 (round-half-away-from-zero):
 *        r = (int64_t)raw[i,j] * (int64_t)scale_mult
 *        half = (scale_shift > 0) ? (1LL << (scale_shift-1)) : 0
 *        q = (r >= 0) ? ((r + half) >> scale_shift)
 *                     : -((-r + half) >> scale_shift)
 *        S[i*N+j] = clamp(q, -128, 127)
 *
 * scale_mult: int32 (positive).
 * scale_shift: int (>=0).
 * M=16, N=16, D=130 (D has a non-128 tail).
 */
void candidate_kernel(const int8_t *Q, const int8_t *K, int8_t *S,
                      int M, int N, int D,
                      int32_t scale_mult, int scale_shift);
#endif /* KERNEL_API_H */
