#ifndef KERNEL_API_H
#define KERNEL_API_H
#include <stdint.h>
/* int8 1x1 (pointwise) convolution on the HMX matrix engine.
 * Input NCHW uint8 [C_in][P] (P = H*W spatial positions), weights int8
 * [C_out][C_in], output [C_out][P] as the uint16 12-bit HMX requant field:
 *   acc[co][p] = sum_ci in[ci*P+p] * W[co*C_in+ci]      (in uint8, W int8)
 *   out[co*P+p] = ((acc*17 + 8) >> 4) & 0xFFF           (0x40-config requant)
 * A 1x1 conv over channels IS a matmul: rows = spatial positions (M=P),
 * reduction = input channels (K=C_in), cols = output channels (N=C_out). The
 * only conv-specific work vs a raw matmul is the NCHW<->position-major gather in
 * the crouton pack. Harness enables the HMX context; use VTCM scratch at
 * HVX_VTCM_BASE. Ranges chosen so the 12-bit requant field is exact. n = P. */
#define CONV_CIN  64
#define CONV_COUT 64
#define CONV_P    64
void candidate_kernel(const uint8_t *in, const int8_t *W, uint16_t *out, int n);
#endif
