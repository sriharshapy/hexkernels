#ifndef KERNEL_API_H
#define KERNEL_API_H
#include <stdint.h>
/* FUSED int8 attention: QK^T THEN AV, two chained HMX matmuls, WITHOUT
 * softmax (linear attention -- documented choice, see prompt.md). S=32 is
 * both the sequence length (query/key/value count) and the head dim D=32
 * (single 32x32 crouton tile per stage, no K-tiling needed).
 *   scores[i][j] = sum_d Q[i*D+d] * K[j*D+d]              (Q uint8 {0,1}, K int8 {0,1})
 *   Sc[i][j]     = sign_extend_12bit((scores*17+8)>>4)      (0x40-config requant; Sc>=0
 *                                                             by construction -> fits uint8,
 *                                                             re-fed as the SECOND matmul's
 *                                                             activation, no softmax)
 *   acc[i][d]    = sum_j Sc[i][j] * V[j*D+d]                (V int8 {-1,0,1})
 *   out[i][d]    = sign_extend_12bit((acc*17+8)>>4)          (0x40-config requant)
 * Q,K,V are [S][D] row-major (S=D=32). Harness enables the HMX context; stage
 * crouton tiles through VTCM at HVX_VTCM_BASE for BOTH matmul stages. Output
 * is bit-exact int32 [S][D].
 */
void candidate_kernel(const uint8_t *Q, const int8_t *K, const int8_t *V,
                       int32_t *out, int S, int D);
#endif
