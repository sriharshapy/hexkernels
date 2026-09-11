#ifndef KERNEL_API_H
#define KERNEL_API_H
#include <stdint.h>
/* Adaptive average pool: reduce a 1D input of n uint8 elements into exactly m
   int8 outputs. The input is divided into m equal-sized contiguous windows
   (n % m == 0), window size k = n/m.
   out[i] = (int8_t)( (in[i*k] + in[i*k+1] + ... + in[i*k+k-1]) / k )
          = integer-truncated mean, then cast to int8 (wraps if mean > 127).
   Large-n / bandwidth-bound: the achievability bar DMA-tiles each window into
   VTCM and reduces the on-chip copies, double-buffered. Here k = 16384 so every
   window is exactly one DMA tile. */
void candidate_kernel(const uint8_t *in, int8_t *out, int n, int m);
#endif
