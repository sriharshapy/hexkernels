#ifndef KERNEL_API_H
#define KERNEL_API_H
#include <stdint.h>
/* Element-wise strict greater-than mask: out[i] = (a[i] > b[i]) ? 255 : 0.
 * a,b are signed int8; out is uint8 (255=true, 0=false). a==b -> 0.
 * Large-N / bandwidth-bound variant: the achievability bar streams DDR<->VTCM
 * via uDMA double-buffering. */
void candidate_kernel(const int8_t *a, const int8_t *b, uint8_t *out, int n);
#endif
