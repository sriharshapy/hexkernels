#ifndef KERNEL_API_H
#define KERNEL_API_H
#include <stdint.h>
/* Sobel gradient magnitude approximation: |Gx| + |Gy|, clamped to [0,255].
   Gx kernel: [[-1,0,1],[-2,0,2],[-1,0,1]]
   Gy kernel: [[-1,-2,-1],[0,0,0],[1,2,1]]
   mag = min(|Gx| + |Gy|, 255)  (L1 approximation of gradient magnitude)
   Border policy: CLAMP-TO-EDGE.
   Output: uint8_t out[w*h], same size as input. */
void candidate_kernel(const uint8_t *in, uint8_t *out, int w, int h);
#endif
