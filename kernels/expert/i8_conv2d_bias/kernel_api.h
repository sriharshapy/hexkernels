#ifndef KERNEL_API_H
#define KERNEL_API_H
#include <stdint.h>
/*
 * Conv2d (3x3, stride=1, SAME padding) + int32 bias + requantize -> int8.
 *
 * Layout: NHWC (batch=1 implicit).
 *   in  : [H][W][C_in]          int8
 *   wt  : [C_out][3][3][C_in]   int8
 *   bias: [C_out]               int32, added after accumulation (per output channel)
 *   out : [H][W][C_out]         int8
 *
 * Computation per out[y][x][co]:
 *   acc    = sum_{ky,kx,ci} in_pad[y+ky-1][x+kx-1][ci] * wt[co][ky][kx][ci]  (int32)
 *   biased = acc + bias[co]   (as int64 to avoid overflow)
 *   Requantize (round-half-away-from-zero):
 *     v    = (int64_t)biased * mult
 *     half = shift > 0 ? (1LL << (shift-1)) : 0
 *     r    = (v >= 0) ? (v+half)>>shift : -(((-v)+half)>>shift)
 *     r   += zp
 *     out[y][x][co] = saturate_to_int8(r)
 *
 * SAME padding: out-of-bounds pixels are zero.
 * mult, shift, zp are runtime params -- do NOT hardcode them.
 */
void candidate_kernel(const int8_t *in, const int8_t *wt, const int32_t *bias,
                      int8_t *out,
                      int H, int W, int C_in, int C_out,
                      int32_t mult, int shift, int8_t zp);
#endif
