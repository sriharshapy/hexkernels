#ifndef KERNEL_API_H
#define KERNEL_API_H
#include <stdint.h>
/* Dilated 1D convolution + requantize: int8 -> int8.
 *
 * Single-channel VALID dilated FIR:
 *   x    : [n + (ntaps-1)*dilation]  int8, caller provides the required context
 *   taps : [ntaps]                   int8 filter coefficients (NOT reversed)
 *   out  : [n]                       int8
 *
 * Computation per output element out[i]:
 *   acc = sum_{k=0}^{ntaps-1} x[i + k*dilation] * taps[k]   (int32 accumulation)
 *   Requantize (round-half-away-from-zero):
 *     v    = (int64_t)acc * mult
 *     half = shift > 0 ? (1LL << (shift-1)) : 0
 *     r    = (v >= 0) ? (v + half) >> shift : -(((-v) + half) >> shift)
 *     r   += zp
 *     out[i] = saturate_to_int8(r)
 *
 * dilation >= 1 (dilation=1 is standard conv1d, dilation=2 skips every other sample).
 * mult, shift, zp are scalar runtime params -- do NOT hardcode them.
 * n=512, ntaps=7, dilation swept {1,2,3}. Pass as runtime args for generality. */
void candidate_kernel(const int8_t *x, const int8_t *taps, int8_t *out,
                      int n, int ntaps, int dilation,
                      int32_t mult, int shift, int8_t zp);
#endif
