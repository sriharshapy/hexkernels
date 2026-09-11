#ifndef KERNEL_API_H
#define KERNEL_API_H
#include <stdint.h>
/* Tile-staged identity copy: out[i] = a[i] for i in [0, n).
 * The achievability bar stages each tile of a[] from DDR into VTCM via uDMA
 * (double-buffered so the next tile's load overlaps the current tile's
 * store), then writes the on-chip copy to out with a plain HVX vector store.
 * This verifies the DMA *load* path moved bytes correctly. */
void candidate_kernel(const int8_t *a, int8_t *out, int n);
#endif
