#ifndef KERNEL_API_H
#define KERNEL_API_H
#include <stdint.h>
/* Sum reduction, int32 accumulator: out[0] = sum_i a[i]. Large-N /
 * bandwidth-bound; the achievability bar DMA-tiles a[] into VTCM and reduces
 * the on-chip copies, double-buffered. */
void candidate_kernel(const int8_t *a, int n, int32_t *out);
#endif
