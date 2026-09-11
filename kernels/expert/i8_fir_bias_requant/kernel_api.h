#ifndef KERNEL_API_H
#define KERNEL_API_H
#include <stdint.h>
/* FIR filter + int32 bias + requantize: int8 -> int8.
 *
 * Single-channel VALID FIR (no zero-padding): x has (n + ntaps - 1) samples.
 *   x    : [n + ntaps - 1]  int8 input samples (extra context pre-loaded by caller)
 *   taps : [ntaps]          int8 filter coefficients (NOT reversed)
 *   bias : scalar int32     single bias added after accumulation
 *   out  : [n]              int8 output
 *
 * Computation per output element out[i]:
 *   acc = sum_{k=0}^{ntaps-1} x[i+k] * taps[k]   (int32 accumulation)
 *   biased = acc + bias
 *   Requantize (round-half-away-from-zero):
 *     v    = (int64_t)biased * mult
 *     half = shift > 0 ? (1LL << (shift-1)) : 0
 *     r    = (v >= 0) ? (v + half) >> shift : -(((-v) + half) >> shift)
 *     r   += zp
 *     out[i] = saturate_to_int8(r)
 *
 * mult, shift, zp are scalar runtime params -- do NOT hardcode them.
 * n=512, ntaps=16. Pass as runtime args for generality. */
void candidate_kernel(const int8_t *x, const int8_t *taps, int32_t bias,
                      int8_t *out,
                      int n, int ntaps,
                      int32_t mult, int shift, int8_t zp);
#endif
