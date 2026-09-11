#ifndef KERNEL_API_H
#define KERNEL_API_H
#include <stdint.h>
/* Argmin reduction: out[0] = index of the minimum element (int8 values).
 * Ties are broken by LOWEST (first) index. Large-N / bandwidth-bound; the
 * achievability bar DMA-streams a[] into VTCM (double-buffered) for the min-value
 * reduction, then locates the first matching index. */
void candidate_kernel(const int8_t *a, int n, int32_t *out);
#endif
