#ifndef KERNEL_API_H
#define KERNEL_API_H
#include <stdint.h>
/* Full MobileNet depthwise-separable block (BN-folded) + requantize, int8->int8.
 *
 * BN has been folded into the depthwise convolution weights and bias (pre-computed
 * by the caller), so the kernel computes:
 *   DW: depthwise 3x3 conv + folded-bias + ReLU (int32 relu)
 *       then requantize DW output to int8
 *   PW: pointwise 1x1 conv on DW int8 output + folded-bias + ReLU (int32)
 *       then requantize PW output to int8
 *
 * Layout: NHWC (batch=1 implicit).
 *   in      : [H][W][C_in]       int8, NHWC row-major
 *   dw_wt   : [C_in][3][3]       int8, depthwise weight (one 3x3 kernel per input channel)
 *   dw_bias : [C_in]             int32, folded BN bias for depthwise stage
 *   pw_wt   : [C_out][C_in]      int8, pointwise 1x1 weight
 *   pw_bias : [C_out]            int32, folded BN bias for pointwise stage
 *   out     : [H][W][C_out]      int8, NHWC row-major
 *
 * Computation:
 *   -- DW stage (per input channel c, SAME zero-padding, stride=1) --
 *   dw_acc[y][x][c] = sum_{ky,kx} in_pad[y+ky-1][x+kx-1][c] * dw_wt[c][ky][kx]  (int32)
 *   dw_biased = dw_acc + dw_bias[c]   (int32)
 *   dw_relu   = max(dw_biased, 0)     (int32 relu)
 *   dw_q[y][x][c]: requantize dw_relu to int8:
 *     v    = (int64_t)dw_relu * dw_mult
 *     half = dw_shift > 0 ? (1LL << (dw_shift-1)) : 0
 *     r    = (v >= 0) ? (v+half) >> dw_shift : -(((-v)+half) >> dw_shift)
 *     r   += dw_zp
 *     dw_q[y][x][c] = saturate_to_int8(r)
 *
 *   -- PW stage (per output channel co) --
 *   pw_acc = sum_{c} dw_q[y][x][c] * pw_wt[co][c]  (int32)
 *   pw_biased = pw_acc + pw_bias[co]   (int32)
 *   pw_relu   = max(pw_biased, 0)      (int32 relu)
 *   out[y][x][co]: requantize pw_relu to int8:
 *     v    = (int64_t)pw_relu * pw_mult
 *     half = pw_shift > 0 ? (1LL << (pw_shift-1)) : 0
 *     r    = (v >= 0) ? (v+half) >> pw_shift : -(((-v)+half) >> pw_shift)
 *     r   += pw_zp
 *     out[y][x][co] = saturate_to_int8(r)
 *
 * dw_mult, dw_shift, dw_zp: requantize params for DW stage (runtime, do NOT hardcode).
 * pw_mult, pw_shift, pw_zp: requantize params for PW stage (runtime, do NOT hardcode).
 * H=8, W=8, C_in=8, C_out=8 (passed as runtime args). */
void candidate_kernel(const int8_t *in,
                      const int8_t *dw_wt, const int32_t *dw_bias,
                      const int8_t *pw_wt, const int32_t *pw_bias,
                      int8_t *out,
                      int H, int W, int C_in, int C_out,
                      int32_t dw_mult, int dw_shift, int8_t dw_zp,
                      int32_t pw_mult, int pw_shift, int8_t pw_zp);
#endif
