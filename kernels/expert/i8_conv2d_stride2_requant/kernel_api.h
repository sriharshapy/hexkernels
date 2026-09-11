#ifndef KERNEL_API_H
#define KERNEL_API_H
#include <stdint.h>
/*
 * Conv2d (3x3, stride=2, SAME padding) + requantize -> int8.
 *
 * Stride-2 halves the spatial resolution: output is ceil(H/2) x ceil(W/2).
 * With SAME padding and stride=2: out_H = (H+1)/2, out_W = (W+1)/2.
 *
 * Layout: NHWC (batch=1 implicit).
 *   in  : [H][W][C_in]              int8
 *   wt  : [C_out][3][3][C_in]       int8
 *   out : [out_H][out_W][C_out]     int8
 *
 * SAME padding for stride=2:
 *   pad_total_h = max(0, (out_H-1)*2 + 3 - H) -- distribute evenly
 *   For simplicity the harness uses H and W where both are even,
 *   so out_H = H/2, out_W = W/2.
 *
 * The padding at each side: pad_top = 1, pad_left = 1 (for f=3, stride=2, even input).
 * So in_pad[y][x][c] accesses in[y-1][x-1][c] with clamp-to-zero.
 *
 * Computation per out[oy][ox][co]  (oy in [0..out_H), ox in [0..out_W)):
 *   iy_base = oy * 2   (stride=2)
 *   ix_base = ox * 2
 *   acc = sum_{ky,kx,ci} in_pad[iy_base+ky-1][ix_base+kx-1][ci] * wt[co][ky][kx][ci]
 *   Requantize (round-half-away-from-zero):
 *     v    = (int64_t)acc * mult
 *     half = shift > 0 ? (1LL << (shift-1)) : 0
 *     r    = (v >= 0) ? (v+half)>>shift : -(((-v)+half)>>shift)
 *     r   += zp
 *     out[oy][ox][co] = saturate_to_int8(r)
 *
 * H, W are the INPUT spatial dims (even). C_in, C_out passed at runtime.
 * mult, shift, zp are runtime params -- do NOT hardcode them.
 */
void candidate_kernel(const int8_t *in, const int8_t *wt,
                      int8_t *out,
                      int H, int W, int C_in, int C_out,
                      int32_t mult, int shift, int8_t zp);
#endif
