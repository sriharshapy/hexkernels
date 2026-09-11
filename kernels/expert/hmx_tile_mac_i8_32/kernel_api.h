#ifndef KERNEL_API_H
#define KERNEL_API_H
#include <stdint.h>
/* Single 32x32x32 int8 tile matmul on the HMX matrix engine (the MAC primitive).
 * One mxclracc, one (activation,weight) mxmem matmul packet, one requant store:
 *   acc[i][j] = sum_{k=0}^{31} A[i*n+k] * B[k*n+j]      (A uint8 0..7, B int8 -3..3)
 *   out[i][j] = sign_extend_12bit( (acc*17 + 8) >> 4 )   (HMX 0x40-config requant
 *                                                          field, as int32)
 * n == 32. The harness enables the HMX context before calling you; stage the
 * activation/weight/output crouton tiles through VTCM at HVX_VTCM_BASE. Output is
 * bit-exact int32 (input range keeps the 12-bit field from wrapping).
 */
void candidate_kernel(const uint8_t *A, const int8_t *B, int32_t *out, int n);
#endif
