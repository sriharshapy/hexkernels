#ifndef KERNEL_API_H
#define KERNEL_API_H
#include <stdint.h>
/* int8 64x64 matmul on the HMX matrix engine with a per-output-column int32
 * bias add FUSED into the HVX epilogue (the classic NPU matmul+bias pattern):
 *   acc[i][j] = sum_k A[i*n+k] * B[k*n+j]         (A uint8 0..7, B int8 -3..3, n=64)
 *   r[i][j]   = sign_extend_12bit((acc*17+8)>>4)  (HMX 0x40-config requant field,
 *                                                   the matrix engine's native output)
 *   out[i][j] = r[i][j] + bias[j]                  (int32, per-output-column bias)
 * The harness enables the HMX context before calling you; use VTCM scratch at
 * HVX_VTCM_BASE for the activation/weight/output crouton tiles. bias is [n] int32.
 * Output is bit-exact int32 (no overflow in this task's input range).
 */
void candidate_kernel(const uint8_t *A, const int8_t *B, const int32_t *bias,
                       int32_t *out, int n);
#endif
