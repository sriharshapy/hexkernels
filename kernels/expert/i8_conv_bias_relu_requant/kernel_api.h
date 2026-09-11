#ifndef KERNEL_API_H
#define KERNEL_API_H
#include <stdint.h>
/* Fused 2D convolution + bias + ReLU + requantize (int8 -> int8).
 *
 * Layout: NHWC (batch=1 implicit, not in shape).
 *   in  : [H][W][C_in]  int8, row-major (C_in is the innermost dimension)
 *   wt  : [C_out][3][3][C_in] int8, row-major
 *   bias: [C_out] int32 (one bias per output channel)
 *   out : [H][W][C_out] int8, row-major
 *
 * Padding: SAME (1 pixel zero-pad on each side for 3x3 kernel, stride=1).
 * So output has the same H x W as input.
 *
 * Computation per output element out[y][x][co]:
 *   acc = sum_{ky,kx,ci} in_pad[y+ky-1][x+kx-1][ci] * wt[co][ky][kx][ci]  (int32)
 *   biased = acc + bias[co]
 *   after_relu = max(biased, 0)
 *   requantize to int8 (round-half-away-from-zero):
 *     v    = (int64_t)after_relu * mult
 *     half = shift > 0 ? (1LL << (shift-1)) : 0
 *     r    = (v >= 0) ? (v + half) >> shift : -(((-v) + half) >> shift)
 *     r   += zp
 *     out[y][x][co] = saturate_to_int8(r)
 *
 * mult, shift, zp are scalar runtime params -- do NOT hardcode them.
 * H=32, W=32, C_in=16, C_out=16 (pass as runtime args for generality). */
void candidate_kernel(const int8_t *in, const int8_t *wt, const int32_t *bias,
                      int8_t *out,
                      int H, int W, int C_in, int C_out,
                      int32_t mult, int shift, int8_t zp);
#endif
