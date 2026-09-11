#ifndef KERNEL_API_H
#define KERNEL_API_H
#include <stdint.h>
/* L3 TILED int8 3x3 conv (VALID) + fused bias + ReLU + saturating requant-to-uint8,
 * lowered to a tiled matmul (im2col) on the HMX matrix engine with DMA/VTCM
 * double-buffered crouton streaming. Composes THREE mechanism groups:
 *   - HMX matrix engine for the im2col matmul (32x32 crouton tiles),
 *   - DMA + VTCM double-buffering to stream each K-tile's crouton DDR->on-chip (the
 *     next K-tile is DMA-prefetched into the alternate VTCM slot while HMX computes
 *     the current one), and
 *   - HVX for the fused bias/ReLU/saturate epilogue.
 *
 * Conv lowered to matmul: M=P (output positions), K=C_in*Kh*Kw (im2col reduction),
 * N=C_out.  in NCHW uint8 [C_in][IH][IW], weights int8 [C_out][C_in*Kh*Kw] (=[C_out][KK]),
 * per-output-channel int32 bias[C_out], output uint8 [C_out][P] (channel-major):
 *   acc[co][p] = sum_{ci,kh,kw} in[ci][oh+kh][ow+kw] * W[co][ci*Kh*Kw + kh*Kw + kw]
 *   r          = sign_extend_12bit((acc*17+8)>>4)   (HMX 0x40-config requant field)
 *   biased     = r + bias[co]
 *   out[co*P+p]= saturate_u8(max(biased, 0))         (fused bias + ReLU + narrow)
 * The harness enables the HMX context AND installs an identity VTCM translation
 * before the timed call so a candidate may DMA DDR<->VTCM and run HMX from VTCM.
 * Input ranges keep |requant result| < 2048 (12-bit field exact). n = P. */
#define C_IN   16
#define C_OUT  64
#define IH     18
#define IW     18
#define KH     3
#define KW     3
#define STRIDE 1
#define OH     ((IH - KH) / STRIDE + 1)   /* 16 */
#define OW     ((IW - KW) / STRIDE + 1)   /* 16 */
#define P_OUT  (OH * OW)                  /* 256 */
#define KK     (C_IN * KH * KW)           /* 144 */
void candidate_kernel(const uint8_t *in, const int8_t *W, const int32_t *bias,
                      uint8_t *out, int n);
#endif
