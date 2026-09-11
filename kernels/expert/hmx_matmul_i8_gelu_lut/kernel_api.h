#ifndef KERNEL_API_H
#define KERNEL_API_H
#include <stdint.h>
/* int8 32x32 matmul + GELU-via-LUT epilogue on the HMX matrix engine.
 *   acc[i][j] = sum_k A[i*n+k] * B[k*n+j]              (A uint8 0..3, B int8 -1..1)
 *   r[i][j]   = sign_extend_12bit((acc*17+8)>>4)       (HMX 0x40-config requant)
 *   out[i][j] = lut[r[i][j] + 128]                     (n=32, 256-entry LUT, int32)
 * The GELU activation is applied by a precomputed 256-entry lookup table indexed
 * by the (small) requantized value + 128; `lut` is provided (identical to the one
 * the reference uses). The harness enables the HMX context; use VTCM scratch at
 * HVX_VTCM_BASE. Output is int32, bit-exact. */
void candidate_kernel(const uint8_t *A, const int8_t *B, const int32_t *lut,
                       int32_t *out, int n);
#endif
