#ifndef KERNEL_API_H
#define KERNEL_API_H
#include <stdint.h>
/* int8 3x3 convolution (VALID) via im2col -> HMX matrix engine.
 * Input NCHW uint8 [C_in][IH][IW], weights int8 [C_out][C_in][KH][KW], output
 * [C_out][P] (P = OH*OW output positions) as the uint16 12-bit HMX requant field:
 *   acc[co][oh][ow] = sum_{ci,kh,kw} in[ci][oh+kh][ow+kw] * W[co][ci][kh][kw]
 *   out[co*P + p]   = ((acc*17 + 8) >> 4) & 0xFFF        (0x40-config requant)
 * Lower the conv to a matmul: rows = output positions (M=P), reduction =
 * (ci,kh,kw) receptive field (K = C_in*KH*KW), cols = output channels (N=C_out).
 * im2col gathers each output position's receptive field into a crouton activation
 * row; the reduction K=72 is zero-padded to 3 crouton K-tiles. Harness enables the
 * HMX context; use VTCM scratch at HVX_VTCM_BASE. n = P. */
#define C_IN   8
#define C_OUT  32
#define IH     10
#define IW     10
#define KH     3
#define KW     3
#define STRIDE 1
#define OH     ((IH - KH) / STRIDE + 1)   /* 8 */
#define OW     ((IW - KW) / STRIDE + 1)   /* 8 */
#define P_OUT  (OH * OW)                  /* 64 */
#define KK     (C_IN * KH * KW)           /* 72 */
void candidate_kernel(const uint8_t *in, const int8_t *W, uint16_t *out, int n);
#endif
