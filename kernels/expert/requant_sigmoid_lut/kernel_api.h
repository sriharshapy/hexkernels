#ifndef KERNEL_API_H
#define KERNEL_API_H
#include <stdint.h>
/* Requantize int32 to uint8 index, then apply sigmoid via a 256-entry LUT.
 * The LUT maps uint8 -> int8 (pre-computed sigmoid values in [-128, 127]).
 * For each i in [0, n):
 *   v    = (int64_t)a[i] * (int64_t)mult
 *   half = shift > 0 ? (1LL << (shift-1)) : 0
 *   r    = (v >= 0) ? (v + half) >> shift : -(((-v) + half) >> shift)
 *   r   += zp
 *   idx  = (uint8_t)clamp(r, 0, 255)    (saturate to uint8 for LUT index)
 *   out[i] = lut[idx]                    (int8 sigmoid output)
 * mult, shift, zp are runtime params -- do NOT hardcode them.
 * lut[] is a runtime 256-entry table -- do NOT hardcode it. */
void candidate_kernel(const int32_t *a, int8_t *out, int n,
                      int32_t mult, int shift, int8_t zp,
                      const int8_t lut[256]);
#endif
