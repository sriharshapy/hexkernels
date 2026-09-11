#ifndef KERNEL_API_H
#define KERNEL_API_H
#include <stdint.h>
/* int8 elementwise add, two's-complement wraparound: out[i]=(int8_t)(a[i]+b[i]).
 * Large-N / bandwidth-bound variant: the achievability bar streams DDR<->VTCM
 * via uDMA double-buffering. */
void candidate_kernel(const int8_t *a, const int8_t *b, int8_t *out, int n);
#endif
