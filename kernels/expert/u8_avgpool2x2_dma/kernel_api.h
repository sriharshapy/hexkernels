#ifndef KERNEL_API_H
#define KERNEL_API_H
#include <stdint.h>
/* 2x2 non-overlapping average pool, stride 2, uint8, integer truncation (no rounding).
 * out[oy*(w/2)+ox] = (in[2oy][2ox] + in[2oy][2ox+1] + in[2oy+1][2ox] + in[2oy+1][2ox+1]) / 4.
 * w and h are even. Window 2x2 / stride 2 / truncate are FIXED constants for this task.
 * Large-image / bandwidth-bound variant: the achievability bar DMA-tiles row blocks
 * of in[] into VTCM (double-buffered) and pools the on-chip copies. */
void candidate_kernel(const uint8_t *in, uint8_t *out, int w, int h);
#endif
