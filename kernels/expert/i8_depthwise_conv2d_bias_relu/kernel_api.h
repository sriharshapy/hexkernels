#ifndef KERNEL_API_H
#define KERNEL_API_H
#include <stdint.h>
/*
 * Depthwise conv2d (3x3, stride=1, SAME padding) + bias + ReLU + requantize -> int8.
 *
 * Layout: NHWC (batch=1 implicit).
 *   in  : [H][W][C]    int8, C is the channel (innermost)
 *   wt  : [C][3][3]    int8, one 3x3 filter PER channel (depthwise)
 *   bias: [C]          int32, per-channel bias added after accumulation
 *   out : [H][W][C]    int8, same spatial size as input
 *
 * Computation per out[y][x][c]:
 *   acc    = sum_{ky,kx} in_pad[y+ky-1][x+kx-1][c] * wt[c][ky][kx]   (int32)
 *   biased = acc + bias[c]
 *   after_relu = max(biased, 0)     <- ReLU clamps negative values to zero
 *   Requantize (round-half-away-from-zero):
 *     v    = (int64_t)after_relu * mult
 *     half = shift > 0 ? (1LL << (shift-1)) : 0
 *     r    = (v >= 0) ? (v+half)>>shift : -(((-v)+half)>>shift)
 *     r   += zp
 *     out[y][x][c] = saturate_to_int8(r)
 *
 * SAME padding: out-of-bounds pixels are zero.
 * mult, shift, zp are runtime params -- do NOT hardcode them.
 */
void candidate_kernel(const int8_t *in, const int8_t *wt, const int32_t *bias,
                      int8_t *out,
                      int H, int W, int C,
                      int32_t mult, int shift, int8_t zp);
#endif
