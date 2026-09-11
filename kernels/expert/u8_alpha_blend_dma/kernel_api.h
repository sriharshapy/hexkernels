#ifndef KERNEL_API_H
#define KERNEL_API_H
#include <stdint.h>
/* uint8 alpha blend: out[i] = (uint8_t)((alpha*a[i] + (256-alpha)*b[i] + 128) >> 8).
 * alpha is a runtime uint16 in [0,256]; alpha=256 -> pure a, alpha=0 -> pure b.
 * Result is naturally in [0,255]. Large-N / bandwidth-bound variant: the
 * achievability bar streams DDR<->VTCM via uDMA double-buffering. Do NOT hardcode. */
void candidate_kernel(const uint8_t *a, const uint8_t *b, uint8_t *out,
                      int n, uint16_t alpha);
#endif
