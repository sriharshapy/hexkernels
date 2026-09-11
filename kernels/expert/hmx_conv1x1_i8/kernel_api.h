#ifndef KERNEL_API_H
#define KERNEL_API_H
#include <stdint.h>
/* int8 1x1 convolution (channel matmul) + per-output-channel bias on the HMX
 * matrix engine.
 *   acc[p][co] = sum_ci In[p*n+ci] * W[co*n+ci]        (In uint8 0..7, W int8 -3..3)
 *   out[p*n+co]= sign_extend_12bit((acc*17+8)>>4) + bias[co]   (n=32, int32 out)
 * In is [P x Cin] activations, W is [Cout x Cin] weights (contract the channel
 * axis -> out = In*W^T), bias is [Cout]. The harness enables the HMX context; use
 * VTCM scratch at HVX_VTCM_BASE. Output is int32, bit-exact. */
void candidate_kernel(const uint8_t *In, const int8_t *W, const int32_t *bias,
                       int32_t *out, int n);
#endif
