#ifndef KERNEL_API_H
#define KERNEL_API_H
#include <stdint.h>
/* Fused fully-connected (linear) + ReLU + requantize -> int8:
 *   acc[i]      = sum_k A[i*K+k] * x[k]         (int32 intermediate)
 *   biased[i]   = acc[i] + bias[i]               (int32)
 *   after_relu  = max(biased[i], 0)              (ReLU)
 *   v           = (int64_t)after_relu * mult
 *   half        = shift > 0 ? (1LL << (shift-1)) : 0
 *   r           = (v >= 0) ? (v+half)>>shift : -(((-v)+half)>>shift)
 *   out[i]      = saturate_i8(r + zp)            (clamp to [-128, 127])
 *
 * A is [M x K] int8 row-major (weight matrix).
 * x is [K] int8 (input vector).
 * bias is [M] int32 (one per output neuron).
 * out is [M] int8.
 * mult, shift, zp are scalar runtime params -- do NOT hardcode them. */
void candidate_kernel(const int8_t *A, const int8_t *x,
                      const int32_t *bias, int8_t *out,
                      int M, int K,
                      int32_t mult, int shift, int8_t zp);
#endif
