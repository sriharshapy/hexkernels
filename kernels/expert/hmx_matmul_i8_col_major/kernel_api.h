#ifndef KERNEL_API_H
#define KERNEL_API_H
#include <stdint.h>
/* int8 32x32 matmul on the HMX matrix engine with COLUMN-MAJOR output.
 *   acc[i][j] = sum_k A[i*n+k] * B[k*n+j]         (A uint8 0..7, B int8 -3..3, n=32)
 *   out[j*n+i] = sign_extend_12bit((acc*17+8)>>4) (COLUMN-MAJOR store: out[j*n+i])
 * The harness enables the HMX context before calling you; use VTCM scratch at
 * HVX_VTCM_BASE for the activation/weight/output crouton tiles. Output is int32,
 * bit-exact (no overflow in this task's range), laid out column-major. */
void candidate_kernel(const uint8_t *A, const int8_t *B, int32_t *out, int n);
#endif
