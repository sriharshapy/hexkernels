#ifndef KERNEL_API_H
#define KERNEL_API_H
#include <stdint.h>
/* L3 tiled int8 GEMM + bias + ReLU + saturating requant-to-uint8 — the canonical
 * on-device NPU inference kernel. Composes THREE mechanism groups:
 *   - HMX matrix engine for the matmul (32x32 crouton tiles),
 *   - DMA + VTCM double-buffering to stream crouton tiles DDR->on-chip (the next
 *     K-tile is DMA-prefetched into the alternate VTCM buffer while HMX computes
 *     the current one), and
 *   - HVX for the fused bias/ReLU/saturate epilogue.
 *
 *   acc[i][j] = sum_k A[i*K+k] * B[k*N+j]           (A uint8 0..3, B int8 -3..3)
 *   r[i][j]   = sign_extend_12bit((acc*17+8)>>4)    (HMX 0x40-config requant field)
 *   biased    = r[i][j] + bias[j]                     (int32 per-output-column bias)
 *   relu      = biased > 0 ? biased : 0
 *   out[i][j] = saturate_u8(relu)                     (clamp to [0,255])
 *
 * M,N,K are multiples of 32 (crouton tile edge). The harness enables the HMX
 * context AND installs an identity VTCM translation before the timed call, so a
 * candidate may both DMA DDR<->VTCM and run HMX from VTCM. Output is bit-exact
 * uint8. bias is [N] int32. Input ranges keep |requant result| < 2048 (12-bit
 * field exact, no HMX saturation). */
void candidate_kernel(const uint8_t *A, const int8_t *B, const int32_t *bias,
                      uint8_t *out, int M, int N, int K);
#endif
