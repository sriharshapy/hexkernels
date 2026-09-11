#ifndef KERNEL_API_H
#define KERNEL_API_H
#include <stdint.h>
/* Two-layer MLP block (fused, integer):
 *   Layer 1:  h[i*V+j]  = gelu_lut[ sat8( sum_k A[i*K+k]*W1[k*V+j] + b1[j] ) ]
 *               sat8(x) = clamp(x, -128, 127)
 *               gelu_lut index = (uint8_t)(sat8(acc+bias) + 128)
 *   Layer 2:  raw[i*D+j]= sum_v h[i*V+v]*W2[v*D+j] + b2[j]   (int32 acc)
 *   Requant:  out[i*D+j] = sat8( round_half_away0( (int64_t)raw * mult >> shift ) + zp )
 *
 * A  : [M x K] int8 row-major (token x input-dim)
 * W1 : [K x V] int8 row-major (input -> hidden)
 * b1 : [V]     int32 (per-hidden bias)
 * gelu_lut : [256] int8 (runtime GELU LUT; index = sat8(acc)+128; do NOT hardcode)
 * W2 : [V x D] int8 row-major (hidden -> output)
 * b2 : [D]     int32 (per-output bias)
 * out: [M x D] int8
 * mult, shift, zp : scalar runtime requant params -- do NOT hardcode
 */
void candidate_kernel(const int8_t  *A,
                      const int8_t  *W1, const int32_t *b1,
                      const int8_t  *gelu_lut,
                      const int8_t  *W2, const int32_t *b2,
                      int8_t        *out,
                      int M, int K, int V, int D,
                      int32_t mult, int shift, int8_t zp);
#endif
