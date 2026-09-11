#ifndef KERNEL_API_H
#define KERNEL_API_H
#include <stdint.h>
/* Two-layer FFN block with ReLU (fused, integer):
 *   Layer 1:  h[i*V+j]  = max(0, sum_k A[i*K+k]*W1[k*V+j] + b1[j])   (ReLU, int32->int8 via sat8)
 *               sat8(x) = clamp(x, -128, 127)
 *               h[i*V+j] = sat8( max(0, acc+b1[j]) )
 *   Layer 2:  raw[i*D+j]= sum_v h[i*V+v]*W2[v*D+j] + b2[j]           (int32 acc)
 *   Requant:  out[i*D+j] = sat8( round_half_away0( (int64_t)raw * mult >> shift ) + zp )
 *
 * A  : [M x K] int8 row-major
 * W1 : [K x V] int8 row-major
 * b1 : [V]     int32
 * W2 : [V x D] int8 row-major
 * b2 : [D]     int32
 * out: [M x D] int8
 * mult, shift, zp: scalar runtime requant params -- do NOT hardcode
 */
void candidate_kernel(const int8_t  *A,
                      const int8_t  *W1, const int32_t *b1,
                      const int8_t  *W2, const int32_t *b2,
                      int8_t        *out,
                      int M, int K, int V, int D,
                      int32_t mult, int shift, int8_t zp);
#endif
