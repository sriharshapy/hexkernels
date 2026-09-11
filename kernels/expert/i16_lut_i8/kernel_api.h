#ifndef KERNEL_API_H
#define KERNEL_API_H
#include <stdint.h>
/* int16 index clamped to [0, lutsize-1] then table lookup -> int8 output.
   idx = clamp(in[i], 0, lutsize-1)
   out[i] = lut[idx]
   lut has lutsize entries; lutsize and the clamp range are RUNTIME params.
   lut supplied as a runtime pointer — do NOT hardcode values. */
void candidate_kernel(const int16_t *in, int8_t *out, int n,
                      const int8_t *lut, int lutsize);
#endif
