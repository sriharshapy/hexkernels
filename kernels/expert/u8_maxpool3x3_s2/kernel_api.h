#ifndef KERNEL_API_H
#define KERNEL_API_H
#include <stdint.h>
/* 3x3 overlapping max pool, stride 2, clamp-to-edge padding.
   Input:  w x h pixels, row-major.
   Output: ow x oh pixels where ow = (w+1)/2, oh = (h+1)/2.
   out[oy*ow+ox] = max of the 3x3 window centred at pixel (2*ox, 2*oy),
   with out-of-bounds accesses clamped to the nearest edge pixel.
   Clamp formula: row = clamp(2*oy+dy, 0, h-1), col = clamp(2*ox+dx, 0, w-1),
   dy, dx ∈ {-1, 0, 1}. */
void candidate_kernel(const uint8_t *in, uint8_t *out, int w, int h);
#endif
