#ifndef KERNEL_API_H
#define KERNEL_API_H
#include <stdint.h>
/*
 * Attention QKᵀ tile, COLUMN-MAJOR K variant: compute scaled dot-product
 * scores S = Q · Kᵀ, requantized to int8.
 *
 * Q: [M x D] int8, row-major        (Q[i,d] = Q[i*D+d]).
 * K: [D x N] int8, COLUMN-major per key -- i.e. K is stored as a [D x N]
 *    matrix, K[d,j] = K[d*N+j].  (NOT [N x D] row-major -- this is the
 *    layout variation vs. the sibling task attention_qkt_tile.)
 * S: [M x N] int8 output (requantized scores).
 *
 * Pinned multi-step integer formula (NO floating point):
 *   1. raw[i,j] = sum_d  Q[i*D+d] * K[d*N+j]           (int32 accumulate,
 *                                                          d = 0..D-1)
 *   2. Requantize raw[i,j] -> int8 (round-half-away-from-zero):
 *        r    = (int64_t)raw[i,j] * (int64_t)scale_mult
 *        half = (scale_shift > 0) ? (1LL << (scale_shift-1)) : 0
 *        q    = (r >= 0) ? ((r + half) >> scale_shift)
 *                        : -((-r + half) >> scale_shift)
 *        S[i*N+j] = clamp(q, -128, 127)
 *
 * scale_mult: int32 (positive), scale_shift: int (>=0) -- both RUNTIME
 * parameters (anti-hardcode; multiple param sets are swept in the harness).
 *
 * M=16, N=16, D=98 (D is NOT a multiple of 128 -- the reduction has a tail
 * when processed in groups of 4).
 */
void candidate_kernel(const int8_t *Q, const int8_t *K, int8_t *S,
                      int M, int N, int D,
                      int32_t scale_mult, int scale_shift);
#endif /* KERNEL_API_H */
