#ifndef KERNEL_API_H
#define KERNEL_API_H
#include <stdint.h>
/* Bitwise-complement computed in VTCM, DMA'd out: out[i] = ~a[i] (bitwise
 * NOT, all 8 bits inverted) for i in [0, n).
 * The achievability bar reads a[] directly from DDR with HVX (no staging on
 * the input side), computes the result into a VTCM scratch buffer, and
 * double-buffers the DMA *store* of that buffer back to DDR so the next
 * tile's compute overlaps the previous tile's DMA-out. This verifies the DMA
 * store path moves the VTCM-computed bytes correctly. */
void candidate_kernel(const int8_t *a, int8_t *out, int n);
#endif
