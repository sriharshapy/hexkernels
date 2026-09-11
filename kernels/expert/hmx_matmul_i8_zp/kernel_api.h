#ifndef KERNEL_API_H
#define KERNEL_API_H
#include <stdint.h>
/* int8 32x32 matmul with a nonzero ACTIVATION ZERO-POINT on the HMX matrix engine.
 *   acc[i][j] = sum_k (A[i*n+k] - zp) * B[k*n+j]        (A uint8 0..7, B int8 -3..3)
 *   out[i][j] = sign_extend_12bit((acc*17+8)>>4)        (n=32, zp small int, int32)
 * The zero-point subtraction distributes as
 *   sum_k (A-zp)*B = sum_k A*B  -  zp * sum_k B[k][j],
 * so it can be realized on HMX as TWO matmuls accumulated (pre-requant) into one
 * accumulator: (A, B) plus (constant zp, -B). The harness enables the HMX
 * context; use VTCM scratch at HVX_VTCM_BASE. Output is int32, bit-exact. */
void candidate_kernel(const uint8_t *A, const int8_t *B, int zp, int32_t *out, int n);
#endif
