#ifndef KERNEL_API_H
#define KERNEL_API_H
#include <stdint.h>
/* int8 32x32 matmul on the HMX matrix engine + per-output-column int32 bias add
 * fused into the epilogue:
 *   acc[i][j] = sum_{k} A[i*n+k] * B[k*n+j]              (A uint8 0..7, B int8 -3..3)
 *   r[i][j]   = sign_extend_12bit( (acc*17 + 8) >> 4 )    (0x40-config HMX requant)
 *   out[i][j] = r[i][j] + bias[j]                          (int32, per-column bias)
 * n == 32; bias is [n] int32. Harness enables the HMX context; stage crouton
 * tiles through VTCM at HVX_VTCM_BASE. Bit-exact int32 output.
 */
void candidate_kernel(const uint8_t *A, const int8_t *B, const int32_t *bias,
                       int32_t *out, int n);
#endif
