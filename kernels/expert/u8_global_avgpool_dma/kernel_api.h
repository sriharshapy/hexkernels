#ifndef KERNEL_API_H
#define KERNEL_API_H
#include <stdint.h>
/* Global average pool over a single flat uint8 plane of n elements.
   out[0] = (uint8_t)( (a[0]+a[1]+...+a[n-1]) / n )   -- int32 accumulator,
   integer division truncates, result cast to uint8.
   Large-n / bandwidth-bound: the achievability bar DMA-tiles a[] into VTCM and
   reduces the on-chip copies, double-buffered. */
void candidate_kernel(const uint8_t *a, int n, uint8_t *out);
#endif
