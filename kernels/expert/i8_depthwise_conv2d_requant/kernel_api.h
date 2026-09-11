#ifndef KERNEL_API_H
#define KERNEL_API_H
#include <stdint.h>
/*
 * Depthwise 2D convolution (3x3, stride=1, SAME padding) + requantize -> int8.
 *
 * Layout: NHWC (batch=1 implicit).
 *   in  : [H][W][C]   int8, C is the channel (innermost)
 *   wt  : [C][3][3]   int8, one 3x3 filter PER channel (depthwise)
 *   out : [H][W][C]   int8, same spatial size as input
 *
 * Computation per out[y][x][c]:
 *   acc = sum_{ky,kx} in_pad[y+ky-1][x+kx-1][c] * wt[c][ky][kx]   (int32)
 *   requantize (round-half-away-from-zero):
 *     v    = (int64_t)acc * mult
 *     half = shift > 0 ? (1LL << (shift-1)) : 0
 *     r    = (v >= 0) ? (v+half)>>shift : -(((-v)+half)>>shift)
 *     r   += zp
 *     out[y][x][c] = saturate_to_int8(r)
 *
 * SAME padding: out-of-bounds pixels are zero.
 * mult, shift, zp are runtime params -- do NOT hardcode them.
 * H, W, C passed at runtime.
 */
void candidate_kernel(const int8_t *in, const int8_t *wt,
                      int8_t *out,
                      int H, int W, int C,
                      int32_t mult, int shift, int8_t zp);
#endif
