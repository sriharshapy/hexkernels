#ifndef KERNEL_API_H
#define KERNEL_API_H
#include <stdint.h>
/* Sum of Absolute Differences: out[0] = sum_i |a[i]-b[i]|, int32 accumulator.
 * Inputs are unsigned bytes. Large-N / bandwidth-bound; the achievability bar
 * DMA-tiles a[] and b[] into VTCM and reduces the on-chip copies, double-buffered. */
void candidate_kernel(const uint8_t *a, const uint8_t *b, int n, int32_t *out);
#endif
