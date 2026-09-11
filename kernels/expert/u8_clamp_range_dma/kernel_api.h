#ifndef KERNEL_API_H
#define KERNEL_API_H
#include <stdint.h>
/* Element-wise clamp to fixed [lo=64, hi=191]:
 *   out[i] = min(max(in[i], 64), 191).
 * in/out are uint8. Large-N / bandwidth-bound variant: the achievability bar
 * streams DDR<->VTCM via uDMA double-buffering. */
void candidate_kernel(const uint8_t *in, uint8_t *out, int n);
#endif
