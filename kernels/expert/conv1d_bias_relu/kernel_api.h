#ifndef KERNEL_API_H
#define KERNEL_API_H
#include <stdint.h>
/* 1D convolution + int32 bias + ReLU -> int8 (VALID padding, multi-channel).
 *
 * Layout (channels-last / interleaved):
 *   x    : [L_in][C]  int8, row-major (C is innermost), L_in = L + K - 1 (no padding)
 *   taps : [K][C]     int8, row-major (one filter bank of K taps, shared across C)
 *   bias : [C]        int32, one bias per channel
 *   out  : [L][C]     int8, row-major
 *
 * Computation per output element out[i][c]:
 *   acc = sum_{k=0}^{K-1} x[i+k][c] * taps[k][c]   (int32 accumulation)
 *   biased = acc + bias[c]
 *   out[i][c] = (int8) sat8(max(biased, 0))   -- ReLU then saturate to int8
 *
 * Saturation: if max(biased,0) > 127 -> 127, < -128 -> -128 (relu guarantees >= 0,
 * so only upper clamp matters; include lower for correctness completeness).
 *
 * L=512, K=5, C=16. Pass as runtime args for generality. */
void candidate_kernel(const int8_t *x, const int8_t *taps, const int32_t *bias,
                      int8_t *out,
                      int L, int K, int C);
#endif
