#ifndef KERNEL_API_H
#define KERNEL_API_H
#include <stdint.h>
/* Global max pool over a single flat int8 plane of n elements (SIGNED max).
   out[0] = max(a[0], a[1], ..., a[n-1])   in [-128, 127].
   Large-n / bandwidth-bound: the achievability bar DMA-tiles a[] into VTCM and
   reduces the on-chip copies, double-buffered. */
void candidate_kernel(const int8_t *a, int n, int8_t *out);
#endif
