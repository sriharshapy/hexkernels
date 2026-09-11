#ifndef KERNEL_API_H
#define KERNEL_API_H
#include <stdint.h>
/* int8 64x64 matmul on the HMX matrix engine with a FULL-MATRIX int32
 * residual add FUSED into the epilogue -- distinct from i8_matmul_add_requant_hmx
 * (which adds an elementwise INT8 residual then saturates to int8): here the
 * residual C is int32 (no int8 saturation anywhere), and the add is a plain
 * elementwise widen-add, not a per-output-column broadcast (unlike the bias
 * sibling task) and not a saturating narrow (unlike the residual-requant sibling):
 *   acc[i][j] = sum_k A[i*n+k] * B[k*n+j]           (A uint8 0..7, B int8 -3..3, n=64)
 *   r[i][j]   = sign_extend_12bit((acc*17+8)>>4)    (HMX 0x40-config requant field)
 *   out[i][j] = r[i][j] + C[i*n+j]                    (int32, no saturation, no narrow)
 * The harness enables the HMX context before calling you; use VTCM scratch at
 * HVX_VTCM_BASE for the activation/weight/output crouton tiles. C is [n*n]
 * int32, row-major, same shape as out. Output is bit-exact int32.
 */
void candidate_kernel(const uint8_t *A, const int8_t *B, const int32_t *C,
                       int32_t *out, int n);
#endif
