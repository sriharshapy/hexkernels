#ifndef KERNEL_API_H
#define KERNEL_API_H
#include <stdint.h>
/* Max reduction, uint8: out[0] = max over i in [0,n) of a[i]. Large-N /
 * bandwidth-bound; the achievability bar DMA-tiles a[] into VTCM and reduces the
 * on-chip copies, double-buffered. */
void candidate_kernel(const uint8_t *a, int n, uint8_t *out);
#endif
