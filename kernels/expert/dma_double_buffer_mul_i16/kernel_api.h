#ifndef KERNEL_API_H
#define KERNEL_API_H
#include <stdint.h>
/* int16 elementwise multiply, low-16-bit truncating (wraparound, matching
 * Q6_Vh_vmpyi_VhVh / C's (int16_t)(a*b)): out[i] = (int16_t)(a[i]*b[i]) for
 * i in [0, n). Large-N / bandwidth-bound. The achievability bar ping-pongs
 * two VTCM buffers for a, b, and out over the N tiles, double-buffered. */
void candidate_kernel(const int16_t *a, const int16_t *b, int16_t *out, int n);
#endif
