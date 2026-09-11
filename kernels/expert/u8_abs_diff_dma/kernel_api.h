#ifndef KERNEL_API_H
#define KERNEL_API_H
#include <stdint.h>
/* uint8 absolute difference: out[i] = (a[i] > b[i]) ? a[i]-b[i] : b[i]-a[i].
 * Large-N / bandwidth-bound variant: the achievability bar streams DDR<->VTCM
 * via uDMA double-buffering. */
void candidate_kernel(const uint8_t *a, const uint8_t *b, uint8_t *out, int n);
#endif
