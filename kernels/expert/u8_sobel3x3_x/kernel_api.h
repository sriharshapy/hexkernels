#ifndef KERNEL_API_H
#define KERNEL_API_H
#include <stdint.h>
/* Sobel x-gradient filter, clamp-to-edge borders.
   Kernel (Sobel-x): [[-1,0,1],[-2,0,2],[-1,0,1]]
   out[y*w+x] = (-1)*in[y-1,x-1] + 0*in[y-1,x] + 1*in[y-1,x+1]
              + (-2)*in[y,x-1]   + 0*in[y,x]   + 2*in[y,x+1]
              + (-1)*in[y+1,x-1] + 0*in[y+1,x] + 1*in[y+1,x+1]
   All out-of-range coords clamped to [0,h-1]/[0,w-1].
   Range of result: [-1020, 1020] — fits in int16.
   Output buffer: int16_t out[w*h], same w*h as input. */
void candidate_kernel(const uint8_t *in, int16_t *out, int w, int h);
#endif
