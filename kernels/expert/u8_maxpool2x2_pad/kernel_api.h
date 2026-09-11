#ifndef KERNEL_API_H
#define KERNEL_API_H
#include <stdint.h>
/* 2x2 max pool, stride 2, ZERO-padding ("same" output size = ceil).
   Output dimensions: ow = (w+1)/2, oh = (h+1)/2, row-major.
   For each output pixel (ox, oy), the 2x2 window reads input pixels at
   (2*oy, 2*ox), (2*oy, 2*ox+1), (2*oy+1, 2*ox), (2*oy+1, 2*ox+1).
   Out-of-bounds pixels are treated as 0 (zero-padding), NOT clamped.
   Equivalent to: pad input with zeros on the right/bottom if needed,
   then non-overlapping 2x2 max with stride 2 on the padded input. */
void candidate_kernel(const uint8_t *in, uint8_t *out, int w, int h);
#endif
