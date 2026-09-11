#ifndef KERNEL_API_H
#define KERNEL_API_H
#include <stdint.h>
/*
 * Conv2d (3x3, stride=1, SAME padding) with folded batchnorm + requantize -> int8.
 *
 * Batchnorm is FOLDED into per-channel scale and shift (already absorbed into weights
 * by the caller). The candidate receives:
 *   - Folded weights: wt[C_out][3][3][C_in] int8 (already bn-scaled)
 *   - Per-channel int32 bias (bn offset term after folding)
 *   - Per-channel scale (bn_scale[co], int32) and shift (bn_shift[co], int)
 *     for the requantize step
 *   - A global zero-point zp applied after per-channel scaling
 *
 * Layout: NHWC.
 *   in      : [H][W][C_in]           int8
 *   wt      : [C_out][3][3][C_in]    int8, folded weights
 *   bias    : [C_out]                int32, folded bn offset
 *   bn_scale: [C_out]                int32, per-channel multiplier
 *   bn_shift: [C_out]                int,   per-channel shift
 *   out     : [H][W][C_out]          int8
 *
 * Computation per out[y][x][co]:
 *   acc    = sum_{ky,kx,ci} in_pad[y+ky-1][x+kx-1][ci] * wt[co][ky][kx][ci]  (int32)
 *   biased = acc + bias[co]
 *   Requantize with per-channel params:
 *     v    = (int64_t)biased * bn_scale[co]
 *     half = bn_shift[co] > 0 ? (1LL << (bn_shift[co]-1)) : 0
 *     r    = (v >= 0) ? (v+half)>>bn_shift[co] : -(((-v)+half)>>bn_shift[co])
 *     r   += zp
 *     out[y][x][co] = saturate_to_int8(r)
 *
 * bn_scale[], bn_shift[], zp are runtime params -- do NOT hardcode them.
 * H, W, C_in, C_out passed at runtime.
 */
void candidate_kernel(const int8_t *in, const int8_t *wt,
                      const int32_t *bias,
                      const int32_t *bn_scale, const int *bn_shift,
                      int8_t zp,
                      int8_t *out,
                      int H, int W, int C_in, int C_out);
#endif
