#ifndef KERNEL_API_H
#define KERNEL_API_H
#include <stdint.h>
/* int8 3x3 convolution (VALID) via im2col -> HMX matrix engine, TILED over a
 * genuinely multi-tile 2D output grid (distinct shape/tiling from
 * i8_conv2d_3x3_hmx: that task is a 2x1 tile grid with 3 K-tiles; this one is
 * a 5x3 tile grid with 3 K-tiles).
 * Input NCHW uint8 [C_in][IH][IW], weights int8 [C_out][C_in][KH][KW], output
 * [C_out][P] (P = OH*OW output positions) as the uint16 12-bit HMX requant field:
 *   acc[co][oh][ow] = sum_{ci,kh,kw} in[ci][oh+kh][ow+kw] * W[co][ci][kh][kw]
 *   out[co*P + p]   = ((acc*17 + 8) >> 4) & 0xFFF        (0x40-config requant)
 * Lower the conv to a matmul: rows = output positions (M=P), reduction =
 * (ci,kh,kw) receptive field (K = C_in*KH*KW), cols = output channels (N=C_out).
 * im2col gathers each output position's receptive field into a crouton activation
 * row; the reduction K is zero-padded to whole crouton K-tiles. P=144 needs a
 * 5-tile M grid, C_out=96 needs a 3-tile N grid -- an explicit 5x3 output-tile
 * loop, NOT a single tile. Harness enables the HMX context; use VTCM scratch at
 * HVX_VTCM_BASE. n = P. */
#define C_IN   8
#define C_OUT  96
#define IH     14
#define IW     14
#define KH     3
#define KW     3
#define STRIDE 1
#define OH     ((IH - KH) / STRIDE + 1)   /* 12 */
#define OW     ((IW - KW) / STRIDE + 1)   /* 12 */
#define P_OUT  (OH * OW)                  /* 144 */
#define KK     (C_IN * KH * KW)           /* 72 */
void candidate_kernel(const uint8_t *in, const int8_t *W, uint16_t *out, int n);
#endif
