#ifndef KERNEL_API_H
#define KERNEL_API_H
#include <stdint.h>
/* 3x3 Laplacian edge filter, clamp-to-edge borders.
   Input: uint8_t image (u8 pixels).
   Kernel: [[0, 1, 0],
            [1,-4, 1],
            [0, 1, 0]]
   out[y*w+x] = in[y-1,x] + in[y+1,x] + in[y,x-1] + in[y,x+1] - 4*in[y,x]
   All out-of-range coords clamped to [0,h-1]/[0,w-1].
   Range: [-4*255, 4*255] = [-1020, 1020], fits in int16_t.
   Output buffer: int16_t out[w*h], same w*h as input. */
void candidate_kernel(const uint8_t *in, int16_t *out, int w, int h);
#endif
