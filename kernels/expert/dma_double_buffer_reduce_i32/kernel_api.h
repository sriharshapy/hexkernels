#ifndef KERNEL_API_H
#define KERNEL_API_H
#include <stdint.h>
/* Sum reduction, int64 accumulator: out[0] = sum_i a[i] for i in [0, n)
 * (int32 inputs, bounded to |a[i]| <= 1000 so a 32-bit VECTOR accumulator
 * never overflows during the loop). Large-N / bandwidth-bound; the
 * achievability bar DMA-tiles a[] into VTCM and reduces the on-chip copies
 * with a 32-lane int32 vector accumulator (Q6_Vw_vadd_VwVw), double-buffered
 * so the next tile's DMA overlaps the current tile's accumulate. Only the
 * FINAL 32-lane-to-scalar reduction widens to int64. */
void candidate_kernel(const int32_t *a, int n, int64_t *out);
#endif
