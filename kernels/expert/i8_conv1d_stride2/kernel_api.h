#ifndef KERNEL_API_H
#define KERNEL_API_H
#include <stdint.h>
/* 1D strided FIR (correlation, stride 2):
   out[i] = sum_{j=0}^{ntaps-1} x[i*stride + j] * taps[j],  i in [0, n).
   x has n*stride + ntaps - 1 samples (VALID window, no padding).
   out has n int32 results. taps NOT reversed. stride is passed at runtime.
   Correlation (not convolution): taps applied in forward order. */
void candidate_kernel(const int8_t *x, const int8_t *taps, int32_t *out,
                      int n, int ntaps, int stride);
#endif
