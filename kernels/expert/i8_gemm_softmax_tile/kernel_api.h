#ifndef KERNEL_API_H
#define KERNEL_API_H
#include <stdint.h>
/*
 * Fused GEMM tile + rowwise softmax: A[M x K] * B[K x N] -> int32 -> softmax -> uint8.
 *
 * Step A -- GEMM:
 *   For each i in [0, M), j in [0, N):
 *     acc[i*N+j] = sum_{k=0}^{K-1} A[i*K+k] * B[k*N+j]   (int32)
 *
 * Step B -- Clamp to int8:
 *   scores[i*N+j] = (int8_t) clamp(acc[i*N+j], -128, 127)
 *
 * Step C -- Rowwise softmax (identical pinned formula):
 *   For each row i in [0, M):
 *     m    = max(scores[i*N+j])
 *     idx  = clamp((int)(scores[i*N+j] - m), -255, 0) + 255
 *     e[j] = exp_lut[idx]                     (uint8, runtime)
 *     S    = sum(e[j])                         (int32)
 *     out[i*N+j] = (uint8)((e[j]*255 + S/2) / S)  (round-half-down)
 *
 * A: [M x K] int8 row-major.
 * B: [K x N] int8 row-major.
 * out: [M x N] uint8.
 * exp_lut: 256 uint8 entries (runtime, opaque).
 * M=GM, N=GN, K=GK are compile-time constants defined below.
 */
#define GM 16
#define GN 16
#define GK 16
void candidate_kernel(const int8_t *A, const int8_t *B,
                      uint8_t *out,
                      const uint8_t *exp_lut);
#endif /* KERNEL_API_H */
