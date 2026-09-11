#ifndef KERNEL_API_H
#define KERNEL_API_H
#include <stdint.h>
/* int8 DENSE 1D conv (VALID, cross-channel) + per-channel bias + requant, via
 * im2col -> HMX. Re-route of the compute-bound i8_conv1d_bias_requant_dma FAIL
 * (that op was depthwise and could not use HMX or DMA); generalised here to the
 * matmul-lowerable DENSE cross-channel form the matrix engine accelerates.
 * Input NCL uint8 [C_in][L_in], weights int8 [C_out][C_in][Kw], bias int32[C_out],
 * output int8 [C_out][OL] (OL = L_in - Kw + 1):
 *   acc[co][ol] = sum_{ci,kw} in[ci][ol+kw] * W[co][ci][kw]
 *   r   = sign_extend_12( (acc*17 + 8) >> 4 )          (native HMX 0x40 requant)
 *   out[co*OL + ol] = clamp(r + bias[co], -128, 127)   (fused bias + int8 requant)
 * Lower to a matmul (M=OL, K=C_in*Kw=80, N=C_out). Harness enables HMX; VTCM at
 * HVX_VTCM_BASE. n = OL. */
#define C_IN   16
#define C_OUT  64
#define KW     5
#define L_IN   68
#define OL     (L_IN - KW + 1)   /* 64 */
#define P_OUT  OL                /* 64 output positions */
#define KK     (C_IN * KW)       /* 80 */
void candidate_kernel(const uint8_t *in, const int8_t *W, const int32_t *bias,
                      int8_t *out, int n);
#endif
