#ifndef KERNEL_API_H
#define KERNEL_API_H
#include <stdint.h>
/* int8 32x32 matmul with a LARGE contraction dimension (K == 128) on the HMX
 * matrix engine, with a ReLU FUSED into the requant epilogue. Same deep-K
 * accumulation shape as hmx_matmul_i8_deepk, but the sign-extended requant
 * field is clamped to >=0 before the store:
 *   acc[i][j] = sum_{k=0}^{K-1} A[i*K+k] * B[k*n+j]     (A uint8 0..3, B int8 -1..1)
 *   r[i][j]   = sign_extend_12bit( (acc*17 + 8) >> 4 )   (0x40-config HMX requant field)
 *   out[i][j] = r[i][j] > 0 ? r[i][j] : 0                 (ReLU)
 * n == 32, K == 128. A is row-major [n][K], B is row-major [K][n], out is [n][n].
 * Harness enables the HMX context; stage crouton tiles through VTCM at
 * HVX_VTCM_BASE. Bit-exact int32 output.
 */
void candidate_kernel(const uint8_t *A, const int8_t *B, int32_t *out, int n, int K);
#endif
