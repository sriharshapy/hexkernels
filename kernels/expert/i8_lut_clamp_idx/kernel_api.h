#ifndef KERNEL_API_H
#define KERNEL_API_H
#include <stdint.h>
/* int8->int8 LUT with CLAMPED index.
   idx = clamp((uint8_t)in[i], lo, hi)
   out[i] = lut[idx]
   lut has 256 entries. lo, hi are runtime params (0 <= lo <= hi <= 255).
   lut supplied as a runtime pointer — do NOT hardcode values.
   lo and hi are swept across multiple calls to prevent hardcoding. */
void candidate_kernel(const int8_t *in, int8_t *out, int n,
                      const int8_t *lut, uint8_t lo, uint8_t hi);
#endif
