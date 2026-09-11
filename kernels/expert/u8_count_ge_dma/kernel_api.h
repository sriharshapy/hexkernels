#ifndef KERNEL_API_H
#define KERNEL_API_H
#include <stdint.h>
/* Count elements >= threshold: out[0] = number of i with a[i] >= t (uint8).
 * Large-N / bandwidth-bound; the achievability bar DMA-tiles a[] into VTCM and
 * counts the on-chip copies, double-buffered. t is a runtime parameter. */
void candidate_kernel(const uint8_t *a, int n, uint8_t t, int32_t *out);
#endif
