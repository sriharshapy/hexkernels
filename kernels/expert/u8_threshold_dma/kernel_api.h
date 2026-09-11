#ifndef KERNEL_API_H
#define KERNEL_API_H
#include <stdint.h>
/* Binary threshold (inclusive), fixed threshold 128:
 *   out[i] = (in[i] >= 128) ? 255 : 0.
 * in/out are uint8. Large-N / bandwidth-bound variant: the achievability bar
 * streams DDR<->VTCM via uDMA double-buffering. */
void candidate_kernel(const uint8_t *in, uint8_t *out, int n);
#endif
