#ifndef KERNEL_API_H
#define KERNEL_API_H
#include <stdint.h>
/* int8 3x3 conv (VALID) + per-output-channel int32 bias + ReLU, via im2col -> HMX.
 * Input NCHW uint8 [C_in][IH][IW], weights int8 [C_out][C_in][KH][KW], per-channel
 * int32 bias[C_out], output int32 [C_out][P] (P = OH*OW):
 *   acc[co][oh][ow] = sum_{ci,kh,kw} in[ci][oh+kh][ow+kw] * W[co][ci][kh][kw]
 *   r   = sign_extend_12( (acc*17 + 8) >> 4 )            (native HMX 0x40 requant)
 *   out[co*P + p] = max(r + bias[co], 0)                 (fused bias + ReLU epilogue)
 * Lower the conv to a matmul (M=P, K=C_in*KH*KW=72, N=C_out); fuse the bias-add and
 * ReLU into the HVX unpack. Harness enables HMX; use VTCM scratch at HVX_VTCM_BASE. n = P. */
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
void candidate_kernel(const uint8_t *in, const int8_t *W, const int32_t *bias,
                      int32_t *out, int n);
#endif
