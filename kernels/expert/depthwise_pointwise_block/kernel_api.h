#ifndef KERNEL_API_H
#define KERNEL_API_H
#include <stdint.h>
/*
 * MobileNet-style depthwise-separable block:
 *   1) Depthwise conv2d (3x3, stride=1, SAME padding, C_in channels)
 *   2) Pointwise conv2d (1x1, C_in -> C_out channels)
 *   Both fused with requantize -> int8 at each stage.
 *
 * Layout: NHWC (batch=1 implicit).
 *   in     : [H][W][C_in]          int8
 *   dw_wt  : [C_in][3][3]          int8, depthwise weights (one 3x3 per input channel)
 *   pw_wt  : [C_out][C_in]         int8, pointwise weights
 *   mid    : [H][W][C_in]          int8, intermediate (after depthwise+requant)
 *   out    : [H][W][C_out]         int8, final output
 *
 * Stage 1 -- depthwise + requant (mid):
 *   acc_dw[y][x][c] = sum_{ky,kx} in_pad[y+ky-1][x+kx-1][c] * dw_wt[c][ky][kx]
 *   mid[y][x][c] = requant(acc_dw, dw_mult, dw_shift, dw_zp)
 *
 * Stage 2 -- pointwise + requant (out):
 *   acc_pw[y][x][co] = sum_{ci} mid[y][x][ci] * pw_wt[co][ci]
 *   out[y][x][co] = requant(acc_pw, pw_mult, pw_shift, pw_zp)
 *
 * requant(v, mult, shift, zp):
 *   r = round_half_away_zero((int64_t)v * mult, shift) + zp
 *   saturate to int8
 *
 * All mult/shift/zp are runtime params -- do NOT hardcode them.
 * mid buffer is provided by the harness (scratch space).
 */
void candidate_kernel(const int8_t *in,
                      const int8_t *dw_wt,
                      const int8_t *pw_wt,
                      int8_t *mid, int8_t *out,
                      int H, int W, int C_in, int C_out,
                      int32_t dw_mult, int dw_shift, int8_t dw_zp,
                      int32_t pw_mult, int pw_shift, int8_t pw_zp);
#endif
