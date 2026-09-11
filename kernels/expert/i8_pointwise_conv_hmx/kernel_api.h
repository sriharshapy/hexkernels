#ifndef KERNEL_API_H
#define KERNEL_API_H
#include <stdint.h>
/* int8 pointwise (1x1) convolution on the HMX matrix engine — deep-channel variant.
 * Distinct from i8_conv2d_1x1_hmx by a deeper channel reduction (C_in=96 = 3 crouton
 * K-tiles) and a wider output-channel count relative to positions. Input NCHW uint8
 * [C_in][P], weights int8 [C_out][C_in], output [C_out][P] as the uint16 12-bit HMX
 * requant field:
 *   acc[co][p]  = sum_ci in[ci*P+p] * W[co*C_in+ci]      (in uint8, W int8)
 *   out[co*P+p] = ((acc*17 + 8) >> 4) & 0xFFF            (0x40-config requant)
 * Pointwise conv over channels IS a matmul: M=P, K=C_in, N=C_out. Harness enables
 * the HMX context; use VTCM scratch at HVX_VTCM_BASE. n = P. */
#define CONV_CIN  96
#define CONV_COUT 64
#define CONV_P    64
void candidate_kernel(const uint8_t *in, const int8_t *W, uint16_t *out, int n);
#endif
