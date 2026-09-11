#ifndef KERNEL_API_H
#define KERNEL_API_H
#include <stdint.h>
/* Sum reduction, int32 in / int32 out: out[0] = sum_i a[i] for i in [0, n).
 * Large-N / bandwidth-bound. The achievability bar DMA-stages tiles of a[]
 * into VTCM (double-buffered) and reduces the on-chip copies with HVX
 * vector add-accumulate, hiding DDR latency behind the reduction. */
void candidate_kernel(const int32_t *a, int n, int32_t *out);
#endif
