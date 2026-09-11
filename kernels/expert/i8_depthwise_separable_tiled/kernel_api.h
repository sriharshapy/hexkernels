#ifndef KERNEL_API_H
#define KERNEL_API_H
#include <stdint.h>
/* L3 TILED depthwise-separable conv block: 3x3 depthwise (per-channel, HVX) followed
 * by a 1x1 pointwise (= dense matmul, HMX) with DMA/VTCM double-buffered streaming of
 * the pointwise activation tiles.  Composes THREE mechanism groups:
 *   - HVX for the per-channel 3x3 depthwise (does NOT lower to a dense matmul),
 *   - HMX matrix engine for the 1x1 pointwise (M=P positions, K=C_in, N=C_out), and
 *   - DMA + VTCM double-buffering to stream the pointwise activation croutons on-chip.
 *
 * NHWC layout (batch=1), VALID conv (no padding):
 *   in   : [IH][IW][C_in]        uint8 (0..3), channel innermost
 *   Wdw  : [C_in][Kh*Kw]         int8  (-1..1), one 3x3 filter PER input channel
 *   Wpw  : [C_out][C_in]         int8  (-1..1), 1x1 pointwise weights
 *   bias : [C_out]               int32, per-output-channel
 *   out  : [P][C_out]            uint8, P = OH*OW output positions (NHWC)
 *
 *   dw[p][c]   = saturate_u8( max( sum_{kh,kw} in[oh+kh][ow+kw][c] * Wdw[c][kh][kw], 0 ) )
 *   acc[p][co] = sum_c dw[p][c] * Wpw[co][c]
 *   r          = sign_extend_12bit((acc*17+8)>>4)     (HMX 0x40-config requant field)
 *   out[p][co] = saturate_u8( max(r + bias[co], 0) )  (fused bias + ReLU + narrow)
 *
 * The harness enables the HMX context AND installs an identity VTCM translation.
 * Input ranges keep dw in 0..27 and |acc| <= 1728 so |r| < 2048 (12-bit field exact).
 * n = P. */
#define C_IN   64
#define C_OUT  64
#define IH     18
#define IW     18
#define KH     3
#define KW     3
#define STRIDE 1
#define OH     ((IH - KH) / STRIDE + 1)   /* 16 */
#define OW     ((IW - KW) / STRIDE + 1)   /* 16 */
#define P_OUT  (OH * OW)                  /* 256 */
void candidate_kernel(const uint8_t *in, const int8_t *Wdw, const int8_t *Wpw,
                      const int32_t *bias, uint8_t *out, int n);
#endif
