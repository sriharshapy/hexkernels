#ifndef KERNEL_API_H
#define KERNEL_API_H
#include <stdint.h>
/* int8 64x64 matmul on the HMX matrix engine with an elementwise residual add
 * + saturating requant-to-int8 FUSED into the HVX epilogue (the classic
 * quantized residual-connection pattern):
 *   acc[i][j] = sum_k A[i*n+k] * B[k*n+j]           (A uint8 0..7, B int8 -3..3, n=64)
 *   r[i][j]   = sign_extend_12bit((acc*17+8)>>4)    (HMX 0x40-config requant field)
 *   v         = r[i][j] + residual[i][j]             (int, residual is int8, elementwise)
 *   out[i][j] = saturate_i8(v)                        (clamp to [-128,127])
 * The harness enables the HMX context before calling you; use VTCM scratch at
 * HVX_VTCM_BASE for the activation/weight/output crouton tiles. residual is
 * [n*n] int8, row-major, same shape as out.
 * Output is bit-exact int8 (saturating narrow).
 */
void candidate_kernel(const uint8_t *A, const int8_t *B, const int8_t *residual,
                       int8_t *out, int n);
#endif
