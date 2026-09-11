#ifndef KERNEL_API_H
#define KERNEL_API_H
#include <stdint.h>
/* Dot product, int32 accumulator: out[0] = sum_i a[i]*b[i] (int16 inputs).
 * Large-N / bandwidth-bound; the achievability bar DMA-tiles a[] and b[] into
 * VTCM and reduces the on-chip copies, double-buffered. Inputs are bounded so
 * the exact int32 sum does not overflow. */
void candidate_kernel(const int16_t *a, const int16_t *b, int n, int32_t *out);
#endif
