#ifndef KERNEL_API_H
#define KERNEL_API_H
#include <stdint.h>
/* Fused depthwise 1D conv + bias + ReLU, int8->int8.
 *
 * Each of C independent channels has its own set of taps.
 * The convolution is a correlation (NOT reversed taps).
 *
 * Input layout:  x[ch * (L + ntaps - 1) + i],  ch in [0,C), i in [0, L+ntaps-1)
 * Taps layout:   taps[ch * ntaps + j],           ch in [0,C), j in [0, ntaps)
 * Bias layout:   bias[ch],                        ch in [0,C)
 * Output layout: out[ch * L + i],                 ch in [0,C), i in [0, L)
 *
 * Computation per output element out[ch][i]:
 *   acc      = sum_{j=0}^{ntaps-1} x[ch*(L+ntaps-1)+i+j] * taps[ch*ntaps+j]  (int32)
 *   biased   = acc + bias[ch]    (int32; bias broadens per-channel)
 *   after_relu = max(biased, 0)  (int32 relu, clamping negatives to zero)
 *   saturate to int8:
 *     r = clamp(after_relu, -128, 127)
 *     out[ch*L + i] = (int8_t)r
 *
 * Taps are NOT reversed; bias is int32 per channel; relu is int32 before saturation.
 * L=256, C=8, ntaps=7 (passed as runtime args for generality). */
void candidate_kernel(const int8_t *x, const int8_t *taps, const int32_t *bias,
                      int8_t *out, int L, int C, int ntaps);
#endif
