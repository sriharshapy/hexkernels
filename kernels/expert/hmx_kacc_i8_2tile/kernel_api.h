#ifndef KERNEL_API_H
#define KERNEL_API_H
#include <stdint.h>
/* HMX int8 K-accumulation over 2 contraction tiles (the accumulate primitive).
 * Output is a single 32x32 tile but the contraction K = 64 = 2*32, so the HMX
 * accumulator must persist across TWO matmul packets:
 *   acc[i][j] = sum_{k=0}^{K-1} A[i*K+k] * B[k*n+j]     (A uint8, B int8; n=32, K=64)
 *   out[i][j] = sign_extend_12bit( (acc*17 + 8) >> 4 )   (0x40-config requant, int32)
 * Issue ONE mxclracc, then for each of the K/32 tiles pack the (activation,weight)
 * pair and issue a matmul packet (they accumulate), then ONE requant store.
 * A is row-major [n][K], B is row-major [K][n], out is [n][n]. Harness enables the
 * HMX context; stage crouton tiles through VTCM at HVX_VTCM_BASE.
 */
void candidate_kernel(const uint8_t *A, const int8_t *B, int32_t *out, int n, int K);
#endif
