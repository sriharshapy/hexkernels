#ifndef KERNEL_API_H
#define KERNEL_API_H
#include <stdint.h>
/* int8 elementwise add, two's-complement wraparound: out[i]=(int8_t)(a[i]+b[i])
 * for i in [0, n). Large-N / DDR-bandwidth-bound. The achievability bar rolls
 * a continuous L2 prefetch (l2fetch) ahead of BOTH input streams a[] and b[]
 * throughout the whole loop (re-issued every 2KB, not just once), combined
 * with loop unrolling for more in-flight loads. */
void candidate_kernel(const int8_t *a, const int8_t *b, int8_t *out, int n);
#endif
