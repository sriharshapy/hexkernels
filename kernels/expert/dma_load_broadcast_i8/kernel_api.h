#ifndef KERNEL_API_H
#define KERNEL_API_H
#include <stdint.h>
/* Broadcast-add a small (exactly 128-byte) operand, cyclically, over a large
 * stream, saturating to int8:
 *   out[i] = clamp(a[i] + op[i % 128], -128, 127)   for i in [0, n)
 * op[] is EXACTLY 128 bytes (one HVX vector) -- load it ONCE and reuse it
 * unchanged for every 128-byte chunk of a[] (no per-chunk reload / no modulo
 * needed inside the vector loop).
 *
 * n is large (working set exceeds L2) so a[] is DDR-bandwidth-bound while op[]
 * is tiny and DDR-latency-free after the first load. To go fast, DMA
 * double-buffer a[] through VTCM in tiles (prefetch tile c+1 while computing
 * tile c) so DDR latency on the large stream is hidden behind compute; op[]
 * only needs to be fetched once, not per tile.
 */
void candidate_kernel(const int8_t *a, const int8_t *op, int8_t *out, int n);
#endif
