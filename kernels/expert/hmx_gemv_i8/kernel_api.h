#ifndef KERNEL_API_H
#define KERNEL_API_H
#include <stdint.h>
/* int8 matrix-vector product (GEMV, N=1) on the HMX matrix engine.
 *   acc[i] = sum_k A[i*n+k] * x[k]                (A uint8 0..7, x int8 -3..3, n=32)
 *   out[i] = sign_extend_12bit((acc*17+8)>>4)     (n=32, int32 out[n])
 * The matrix engine is a 32x32 tile; a GEMV uses a single weight column (x in
 * column 0, other columns zero) and reads output column 0. The harness enables
 * the HMX context; use VTCM scratch at HVX_VTCM_BASE. Output is int32, bit-exact. */
void candidate_kernel(const uint8_t *A, const int8_t *x, int32_t *out, int n);
#endif
