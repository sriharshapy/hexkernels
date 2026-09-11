#ifndef KERNEL_API_H
#define KERNEL_API_H
#include <stdint.h>
/* int8 32x32 matmul with a CHAINED K=64 accumulation on the HMX matrix engine.
 *   acc[i][j] = sum_{k=0}^{k_dim-1} A[i*k_dim+k] * B[k*n+j]  (A uint8 0..7, B int8 -3..3)
 *   out[i][j] = sign_extend_12bit((acc*17+8)>>4)             (n=32, k_dim=64, int32)
 * K=64 is chained over two 32-deep K-tiles: clear the HMX accumulator ONCE
 * (mxclracc), issue BOTH (activation,weight) matmul packets so they accumulate,
 * then a single requant store. The harness enables the HMX context; use VTCM
 * scratch at HVX_VTCM_BASE. Output is int32, bit-exact. */
void candidate_kernel(const uint8_t *A, const int8_t *B, int32_t *out, int n, int k_dim);
#endif
