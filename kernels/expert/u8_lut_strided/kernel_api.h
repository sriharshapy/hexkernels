#ifndef KERNEL_API_H
#define KERNEL_API_H
#include <stdint.h>
/* Apply a 256-entry uint8->uint8 LUT to every k-th element; copy all others unchanged.
   for i in [0,n):
     if (i % k == 0):  out[i] = lut[in[i]]
     else:             out[i] = in[i]
   lut has 256 entries supplied as a runtime pointer.
   k is a RUNTIME param (swept to prevent hardcoding). */
void candidate_kernel(const uint8_t *in, uint8_t *out, int n,
                      const uint8_t *lut, int k);
#endif
