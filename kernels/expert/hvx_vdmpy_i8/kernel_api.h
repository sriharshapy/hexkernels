#ifndef KERNEL_API_H
#define KERNEL_API_H
#include <stdint.h>
/*
 * Dual unsigned-byte x signed-byte multiply-accumulate to int16, g=262
 * groups (tail path: 262 groups = 8*32 + 6; input length n = 4*g = 1048
 * uint8 elements; output length 2*g = 524 int16 elements).
 *
 * Fixed 4-byte signed weight w[4] = {w0,w1,w2,w3} is shared across ALL
 * groups (broadcast, like the HVX scalar-register operand of
 * Q6_Vh_vdmpy_VubRb). For group k (input bytes a[4k..4k+3], all uint8):
 *   out[2*k]   = (int16_t)((int)a[4*k+0]*w0 + (int)a[4*k+1]*w1)
 *   out[2*k+1] = (int16_t)((int)a[4*k+2]*w2 + (int)a[4*k+3]*w3)
 * Two's-complement WRAP on overflow (the hardware MAC does not saturate).
 * Matches Q6_Vh_vdmpy_VubRb with Rt packed little-endian as
 * (w3<<24)|(w2<<16)|(w1<<8)|w0.
 */
void candidate_kernel(const uint8_t *a, const int8_t w[4], int16_t *out, int g);
#endif
