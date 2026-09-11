#ifndef KERNEL_API_H
#define KERNEL_API_H
#include <stdint.h>
/* Top-1 reduction: returns BOTH the maximum value and its index.
 *   out[0] = value of the maximum element (as int32).
 *   out[1] = index of the maximum element; on ties the lowest (first) index wins.
 * Large-N / bandwidth-bound; the achievability bar DMA-streams a[] into VTCM
 * (double-buffered) for the max-value reduction, then locates the first index. */
void candidate_kernel(const int8_t *a, int n, int32_t *out);
#endif
