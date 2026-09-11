#ifndef KERNEL_API_H
#define KERNEL_API_H
#include <stdint.h>
/* Integer variance: out[0] = (n*sum(a[i]^2) - (sum a[i])^2) / n, integer floor.
 * All intermediates fit in int64. Large-N / bandwidth-bound; the achievability
 * bar DMA-tiles a[] into VTCM and reduces the on-chip copies, double-buffered. */
void candidate_kernel(const int8_t *a, int n, int32_t *out);
#endif
