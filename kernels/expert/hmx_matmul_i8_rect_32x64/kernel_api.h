#ifndef KERNEL_API_H
#define KERNEL_API_H
#include <stdint.h>
/* RECTANGULAR int8 matmul on the HMX matrix engine: M=32, N=64, K=32. The output
 * is 32x64 = a 1x2 grid of 32x32 output tiles (one per 32-column block of N):
 *   acc[i][j] = sum_{k=0}^{K-1} A[i*K+k] * B[k*N+j]     (A uint8 0..7, B int8 -3..3)
 *   out[i][j] = sign_extend_12bit( (acc*17 + 8) >> 4 )   (0x40-config HMX requant)
 * A is row-major [M][K], B is row-major [K][N], out is [M][N]. Harness enables the
 * HMX context; stage crouton tiles through VTCM at HVX_VTCM_BASE. Bit-exact int32.
 */
void candidate_kernel(const uint8_t *A, const int8_t *B, int32_t *out,
                       int M, int N, int K);
#endif
