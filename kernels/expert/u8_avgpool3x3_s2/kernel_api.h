#ifndef KERNEL_API_H
#define KERNEL_API_H
#include <stdint.h>
/* 3x3 overlapping average pool, stride 2, clamp-to-edge padding.
   Input:  w x h pixels, row-major.
   Output: ow x oh pixels where ow = (w+1)/2, oh = (h+1)/2.
   out[oy*ow+ox] = floor( sum of 9 clamped-border pixels / 9 )  (integer truncation).
   Centre pixel: (2*ox, 2*oy).  Out-of-bounds: clamp to nearest edge. */
void candidate_kernel(const uint8_t *in, uint8_t *out, int w, int h);
#endif
