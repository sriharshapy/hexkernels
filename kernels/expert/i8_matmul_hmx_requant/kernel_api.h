#ifndef KERNEL_API_H
#define KERNEL_API_H
#include <stdint.h>
/* int8 32x32 matmul on the HMX matrix engine, REQUANTIZED DOWN TO int8 (not the
 * raw 12-bit/uint16 field the sibling matmul tasks expose):
 *   acc[i][j]  = sum_k A[i*N+k] * B[k*N+j]      (A uint8 0..3, B int8 -1..1, N=32)
 *   r          = (acc*17 + 8) >> 4              (0x40-config requant: scale 17/16, bias 0)
 *   out[i*N+j] = (int8_t) r                     (input ranges chosen so |r| <= 127:
 *                                                 no int8 saturation, cast is exact)
 * A realistic requantize-to-int8 NPU epilogue: activation is unsigned (HMX only
 * offers activation.ub on v68), weight is signed int8 -- the same asymmetric
 * quantization convention real int8 NPU kernels use.
 * The harness enables the HMX context before calling you; use VTCM scratch at
 * HVX_VTCM_BASE for the activation/weight/output crouton tiles.
 */
void candidate_kernel(const uint8_t *A, const int8_t *B, int8_t *out, int n);
#endif
