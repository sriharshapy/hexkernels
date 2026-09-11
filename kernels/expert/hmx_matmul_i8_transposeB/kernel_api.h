#ifndef KERNEL_API_H
#define KERNEL_API_H
#include <stdint.h>
/* int8 32x32 matmul on the HMX matrix engine with a PRE-TRANSPOSED weight (B^T).
 * B arrives transposed as Bt[j*n+k] = B_logical[k][j], so
 *   acc[i][j] = sum_k A[i*n+k] * Bt[j*n+k]        (A uint8 0..7, Bt int8 -3..3, n=32)
 *   out[i][j] = sign_extend_12bit((acc*17+8)>>4)  (HMX 0x40-config requant field)
 * The harness enables the HMX context before calling you; use VTCM scratch at
 * HVX_VTCM_BASE for the activation/weight/output crouton tiles. Output is int32
 * and bit-exact (no overflow in this task's input range). */
void candidate_kernel(const uint8_t *A, const int8_t *Bt, int32_t *out, int n);
#endif
