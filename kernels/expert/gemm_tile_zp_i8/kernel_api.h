#ifndef KERNEL_API_H
#define KERNEL_API_H
#include <stdint.h>
/* Zero-point-quantized int8 GEMM tile -> requantized int8 output.
 *
 * A is [M x K] int8 row-major, a quantized ACTIVATION with a runtime int32
 * zero-point zpA to subtract (A[i,k] = A[i*K+k]). B is [K x N] int8
 * row-major, CONVENTIONAL layout (B[k,j] = B[k*N+j]), the WEIGHT -- already
 * zero-centered, no zero-point needed on B. out is [M x N] int8.
 *
 * Standard zero-point GEMM decomposition (do NOT subtract zpA from A before
 * multiplying -- that can push A[i,k]-zpA outside the int8 range and
 * overflow; decompose the subtraction algebraically instead):
 *   rawdot[i,j] = sum_k A[i*K+k] * B[k*N+j]        (int32; plain int8xint8)
 *   colsum[j]   = sum_k B[k*N+j]                    (int32; shared over i)
 *   acc[i,j]    = rawdot[i,j] - zpA * colsum[j]
 *   r    = (int64_t)acc[i,j] * (int64_t)scale_mult
 *   half = (scale_shift > 0) ? (1LL << (scale_shift-1)) : 0
 *   q    = (r >= 0) ? ((r + half) >> scale_shift) : -((-r + half) >> scale_shift)
 *   out[i*N+j] = clamp(q, -128, 127)
 *
 * zpA, scale_mult, scale_shift are RUNTIME parameters (anti-hardcode;
 * multiple param triples are swept, including zpA=0). M=8, N=8, K=94
 * (94 = 23*4 + 2 is NOT a multiple of 4 -- reduction tail).
 */
void candidate_kernel(const int8_t *A, const int8_t *B, int8_t *out,
                      int M, int N, int K,
                      int32_t zpA, int32_t scale_mult, int scale_shift);
#endif /* KERNEL_API_H */
