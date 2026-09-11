#ifndef KERNEL_API_H
#define KERNEL_API_H
#include <stdint.h>
/* 1D FIR (correlation, taps NOT reversed): out[i] = sum_{j=0}^{ntaps-1} x[i+j]*taps[j].
 * x has n+ntaps-1 samples; out has n int32 results. ntaps is fixed to 8.
 * Large-N / bandwidth-bound (int32 output dominates traffic); the achievability
 * bar DMA-tiles x[] (with halo) into VTCM and streams outputs back, double-buffered. */
void candidate_kernel(const int8_t *x, const int8_t *taps, int32_t *out, int n, int ntaps);
#endif
