#ifndef KERNEL_API_H
#define KERNEL_API_H
#include <stdint.h>
/* int8 64x64 matmul on the HMX matrix engine with a leaky-ReLU FUSED into the
 * HVX epilogue (no bias):
 *   acc[i][j] = sum_k A[i*n+k] * B[k*n+j]          (A uint8 0..7, B int8 -3..3, n=64)
 *   r[i][j]   = sign_extend_12bit((acc*17+8)>>4)   (HMX 0x40-config requant field)
 *   out[i][j] = r[i][j] >= 0 ? r[i][j] : (r[i][j] >> 3)   (leaky ReLU, slope 1/8)
 * The harness enables the HMX context before calling you; use VTCM scratch at
 * HVX_VTCM_BASE for the activation/weight/output crouton tiles.
 * Output is bit-exact int32 (>>3 is an arithmetic right shift on the signed
 * value, matching the scalar reference exactly).
 */
void candidate_kernel(const uint8_t *A, const int8_t *B, int32_t *out, int n);
#endif
