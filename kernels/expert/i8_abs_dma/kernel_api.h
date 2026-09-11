#ifndef KERNEL_API_H
#define KERNEL_API_H
#include <stdint.h>
/* int8 elementwise saturating absolute value:
 *   out[i] = (a[i] == -128) ? 127 : (a[i] < 0 ? -a[i] : a[i])
 * Large-N / bandwidth-bound variant: the achievability bar streams DDR<->VTCM
 * via uDMA double-buffering. */
void candidate_kernel(const int8_t *a, int8_t *out, int n);
#endif
