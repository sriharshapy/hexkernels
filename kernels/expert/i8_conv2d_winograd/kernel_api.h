#ifndef KERNEL_API_H
#define KERNEL_API_H
#include <stdint.h>
/* Conv2d 3x3 + requantize, int8->int8.
 *
 * Layout: NHWC (batch=1 implicit).
 *   in  : [H][W][C_in]       int8, NHWC row-major
 *   wt  : [C_out][3][3][C_in] int8, row-major
 *   out : [H][W][C_out]      int8, NHWC row-major
 *
 * Padding: SAME (1 pixel zero-pad each side for 3x3, stride=1).
 * Output has same H x W as input.
 *
 * Computation per output element out[y][x][co]:
 *   acc = sum_{ky,kx,ci} in_pad[y+ky-1][x+kx-1][ci] * wt[co][ky][kx][ci]  (int32)
 *   requantize to int8:
 *     v    = (int64_t)acc * mult
 *     half = shift > 0 ? (1LL << (shift-1)) : 0
 *     r    = (v >= 0) ? (v + half) >> shift : -(((-v) + half) >> shift)
 *     r   += zp
 *     out[y][x][co] = saturate_to_int8(r)
 *
 * NOTE for candidates: the correctness reference is DIRECT 3x3 convolution.
 * An optimized candidate may use Winograd F(2,3) tiles for speed -- but the
 * output must be bit-exact with the direct-conv reference defined above.
 * mult, shift, zp are scalar runtime params -- do NOT hardcode them.
 * H=12, W=12, C_in=8, C_out=8 (passed as runtime args for generality). */
void candidate_kernel(const int8_t *in, const int8_t *wt, int8_t *out,
                      int H, int W, int C_in, int C_out,
                      int32_t mult, int shift, int8_t zp);
#endif
