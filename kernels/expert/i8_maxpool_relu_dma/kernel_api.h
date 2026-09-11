#ifndef KERNEL_API_H
#define KERNEL_API_H
#include <stdint.h>
/* Fused 2x2 non-overlapping max pool (signed int8) then ReLU (clamp to [0,127]).
 * out[oy*(w/2)+ox] = relu(max(in[2oy][2ox], in[2oy][2ox+1], in[2oy+1][2ox], in[2oy+1][2ox+1])),
 *   relu(x) = x > 0 ? x : 0.
 * w and h are even. Window 2x2 / stride 2 / ReLU are FIXED for this task.
 * Large-image / bandwidth-bound variant: the achievability bar DMA-tiles row blocks
 * of in[] into VTCM (double-buffered) and pools the on-chip copies. */
void candidate_kernel(const int8_t *in, int8_t *out, int w, int h);
#endif
