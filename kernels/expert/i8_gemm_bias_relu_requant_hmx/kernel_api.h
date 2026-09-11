#ifndef KERNEL_API_H
#define KERNEL_API_H
#include <stdint.h>
/* int8 64x64 GEMM on the HMX matrix engine with bias + ReLU + saturating
 * requant-to-uint8 FUSED into the HVX epilogue -- the classic quantized-
 * inference matmul epilogue chain:
 *   acc[i][j] = sum_k A[i*n+k] * B[k*n+j]          (A uint8 0..7, B int8 -3..3, n=64)
 *   r[i][j]   = sign_extend_12bit((acc*17+8)>>4)   (HMX 0x40-config requant field)
 *   biased    = r[i][j] + bias[j]                    (int32, per-output-column bias)
 *   relu      = biased > 0 ? biased : 0
 *   out[i][j] = saturate_u8(relu)                    (clamp to [0,255])
 * The harness enables the HMX context before calling you; use VTCM scratch at
 * HVX_VTCM_BASE for the activation/weight/output crouton tiles. bias is [n] int32.
 * Output is bit-exact uint8 (saturating narrow).
 */
void candidate_kernel(const uint8_t *A, const int8_t *B, const int32_t *bias,
                       uint8_t *out, int n);
#endif
