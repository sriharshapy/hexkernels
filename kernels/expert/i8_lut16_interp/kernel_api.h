#ifndef KERNEL_API_H
#define KERNEL_API_H
#include <stdint.h>
/* 16-entry linearly-interpolated LUT.
   For each input byte b = (uint8_t)in[i]:
     hi = b >> 4          (index into lut, 0..15)
     lo = b & 0xF         (interpolation fraction, 0..15)
   out[i] = (int8_t)(lut[hi] + (((int16_t)(lut[hi+1] - lut[hi]) * lo) >> 4))
   lut has 17 entries (lut[0]..lut[16]) to allow lut[hi+1] when hi=15.
   Truncation (not rounding) is used for the >>4 shift.
   lut supplied as runtime pointer — do NOT hardcode. */
void candidate_kernel(const int8_t *in, int8_t *out, int n, const int8_t *lut);
#endif
