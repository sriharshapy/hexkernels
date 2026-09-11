#ifndef KERNEL_API_H
#define KERNEL_API_H
#include <stdint.h>
/* Toy operand-streaming MAC: n = nblocks*blk. For each block m,
 *   out[m] = sum_{k=0}^{blk-1} (int32)a[m*blk+k] * (int32)b[m*blk+k]
 * where a is uint8 (activations) and b is int8 (weights); accumulate in int32.
 * blk is a multiple of 128. The achievability bar DMA-streams a/b block tiles
 * into VTCM (double-buffered) and MAC-accumulates the on-chip copies. */
void candidate_kernel(const uint8_t *a, const int8_t *b, int32_t *out, int n, int blk);
#endif
