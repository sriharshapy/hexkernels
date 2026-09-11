#ifndef KERNEL_API_H
#define KERNEL_API_H
#include <stdint.h>
/* Global average pool over a single flat int8 plane of n elements.
   out[0] = (int8_t)( (a[0]+a[1]+...+a[n-1]) / n )   -- int32 accumulator,
   integer division truncates toward zero, result cast (wrapped) to int8.
   Large-n / bandwidth-bound: the achievability bar DMA-tiles a[] into VTCM and
   reduces the on-chip copies, double-buffered. */
void candidate_kernel(const int8_t *a, int n, int8_t *out);
#endif
