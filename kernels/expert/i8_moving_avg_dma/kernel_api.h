#ifndef KERNEL_API_H
#define KERNEL_API_H
#include <stdint.h>
/* 1D box moving average (VALID window, no padding), window W fixed to 8:
   out[i] = (int8_t)( sum_{j=0}^{W-1} (int32)x[i+j] / W ),  i in [0,n),
   division truncates toward zero (C integer /). x has n+W-1 samples; out has n.
   Large-N / bandwidth-bound; the achievability bar DMA-tiles x[] (with halo)
   into VTCM and streams outputs back, double-buffered. */
void candidate_kernel(const int8_t *x, int8_t *out, int n, int W);
#endif
