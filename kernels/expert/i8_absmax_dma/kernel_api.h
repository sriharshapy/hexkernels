#ifndef KERNEL_API_H
#define KERNEL_API_H
#include <stdint.h>
/* Absolute-max reduction: out[0] = max over i of |a[i]|; note |-128| = 128.
 * Large-N / bandwidth-bound; the achievability bar DMA-tiles a[] into VTCM and
 * reduces the on-chip copies, double-buffered. */
void candidate_kernel(const int8_t *a, int n, int32_t *out);
#endif
