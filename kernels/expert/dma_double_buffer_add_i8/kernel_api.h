#ifndef KERNEL_API_H
#define KERNEL_API_H
#include <stdint.h>
/* Saturating int8 elementwise add, DMA double-buffered over N tiles:
 * out[i] = clamp(a[i] + b[i], -128, 127)  for i in [0, n).
 * (Saturating, NOT two's-complement wraparound.) The achievability bar
 * ping-pongs two VTCM buffer pairs across the N tiles: while tile c is
 * being added on-chip, tile c+1's inputs are DMA'd in and tile c-1's
 * result is DMA'd out, so DDR latency on both directions is hidden behind
 * compute. */
void candidate_kernel(const int8_t *a, const int8_t *b, int8_t *out, int n);
#endif
