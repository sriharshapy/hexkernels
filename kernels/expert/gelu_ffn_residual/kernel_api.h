#ifndef KERNEL_API_H
#define KERNEL_API_H
#include <stdint.h>
/* FFN with GELU + residual add + LayerNorm (fused, integer):
 *   FFN layer 1: h[i*V+v] = gelu_lut[ sat8( sum_k x[i*K+k]*W1[k*V+v] + b1[v] ) ]
 *   FFN layer 2: ffn[i*D+j] = sum_v h[i*V+v]*W2[v*D+j] + b2[j]   (int32 acc)
 *   Residual:   res[i*D+j] = sat8( ffn[i*D+j] + (int32_t)x_res[i*D+j] )
 *                 (x_res is the skip-connection input, same shape as out [M x D])
 *   LayerNorm per token i (same formula as layernorm_i8):
 *     mu  = mean( res[i*D+0..D-1] )           (integer division)
 *     var = mean( (res[i*D+j] - mu)^2 )       (integer division)
 *     inv = inv_lut[ clamp(var, 0, 255) ]      (uint8; runtime LUT -- do NOT hardcode)
 *     out[i*D+j] = sat8(
 *       (((res[i*D+j]-mu) * gamma[j] + 64) >> 7) * inv + 128) >> 8)
 *       + beta[j] )
 *
 * x      : [M x K] int8 (token input to FFN)
 * W1     : [K x V] int8 row-major
 * b1     : [V]     int32
 * gelu_lut: [256]  int8 (runtime GELU LUT; index = sat8(acc+b1)+128; NOT hardcoded)
 * W2     : [V x D] int8 row-major
 * b2     : [D]     int32
 * x_res  : [M x D] int8 (residual / skip input)
 * gamma  : [D]     int8  (LayerNorm scale)
 * beta   : [D]     int8  (LayerNorm offset)
 * inv_lut: [256]   uint8 (LayerNorm inverse-sqrt LUT; NOT hardcoded)
 * out    : [M x D] int8
 */
void candidate_kernel(const int8_t   *x,
                      const int8_t   *W1, const int32_t *b1,
                      const int8_t   *gelu_lut,
                      const int8_t   *W2, const int32_t *b2,
                      const int8_t   *x_res,
                      const int8_t   *gamma, const int8_t *beta,
                      const uint8_t  *inv_lut,
                      int8_t         *out,
                      int M, int K, int V, int D);
#endif
