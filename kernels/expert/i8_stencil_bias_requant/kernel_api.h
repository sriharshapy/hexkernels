#ifndef KERNEL_API_H
#define KERNEL_API_H
#include <stdint.h>
/* 3x3 stencil convolution + bias + requantize: i8->i8.
 * Input:  int8 image in[h*w].
 * Kernel: int8 weights[9] (row-major 3x3), runtime param.
 * Bias:   int32 bias scalar, added after stencil sum.
 * Requantize: out[y*w+x] = sat_i8(round_half_away_from_zero((acc + bias) * mult >> shift) + zp)
 *   acc = sum_{dy,dx} weights[(dy+1)*3+(dx+1)] * in[clamp(y+dy)*w + clamp(x+dx)]
 *   v = acc + bias
 *   half = shift > 0 ? (1LL << (shift-1)) : 0
 *   r = (v >= 0) ? (v*mult + half) >> shift : -((-v*mult + half) >> shift)
 *   r += zp; clamp to [-128, 127].
 * Border: clamp-to-edge.
 * weights, bias, mult, shift, zp are runtime params -- do NOT hardcode. */
void candidate_kernel(const int8_t *in, int8_t *out, int w, int h,
                      const int8_t *weights, int32_t bias,
                      int32_t mult, int shift, int8_t zp);
#endif
