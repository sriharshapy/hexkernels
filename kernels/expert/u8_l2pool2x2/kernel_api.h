#ifndef KERNEL_API_H
#define KERNEL_API_H
#include <stdint.h>
/* 2x2 L2 pool, stride 2.  Output is (w/2) x (h/2), stored row-major.
   out[oy*(w/2)+ox] = (uint8_t) min(255, (uint32_t) sqrtf(a*a + b*b + c*c + d*d))
   where a,b,c,d are the four uint8 pixels in the 2x2 window:
     a = in[2oy*w+2ox],   b = in[2oy*w+2ox+1],
     c = in[(2oy+1)*w+2ox], d = in[(2oy+1)*w+2ox+1].
   Use standard C sqrtf() and truncate (floor) to uint8, saturating at 255.
   w and h are even. */
void candidate_kernel(const uint8_t *in, uint8_t *out, int w, int h);
#endif
