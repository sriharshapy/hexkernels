#ifndef KERNEL_API_H
#define KERNEL_API_H
#include <stdint.h>
/* Depthwise 1D FIR (correlation): C independent channels, each with its own taps.
   Input layout:  x[ch * (L + ntaps - 1) + i], ch in [0,C), i in [0, L+ntaps-1).
   Taps layout:   taps[ch * ntaps + j],          ch in [0,C), j in [0,ntaps).
   Output layout: out[ch * L + i],                ch in [0,C), i in [0,L).
   out[ch][i] = sum_{j=0}^{ntaps-1} x[ch*(L+ntaps-1)+i+j] * taps[ch*ntaps+j].
   taps NOT reversed. int32 accumulator. */
void candidate_kernel(const int8_t *x, const int8_t *taps, int32_t *out,
                      int L, int C, int ntaps);
#endif
