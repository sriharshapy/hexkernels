#ifndef KERNEL_API_H
#define KERNEL_API_H
#include <stdint.h>
/* Requantize int32 to int8, then clip to [lo, hi].
 * For each i in [0, n):
 *   v    = (int64_t)a[i] * (int64_t)mult
 *   half = shift > 0 ? (1LL << (shift-1)) : 0
 *   r    = (v >= 0) ? (v + half) >> shift : -(((-v) + half) >> shift)
 *   r   += zp
 *   r    = clamp(r, -128, 127)          (int8 saturation)
 *   out[i] = (int8_t)clamp(r, lo, hi)   (application clip, lo >= -128, hi <= 127)
 * mult, shift, zp, lo, hi are runtime params -- do NOT hardcode them. Large-N /
 * bandwidth-bound variant: the achievability bar streams DDR<->VTCM via uDMA
 * double-buffering. */
void candidate_kernel(const int32_t *a, int8_t *out, int n,
                      int32_t mult, int shift, int8_t zp,
                      int8_t lo, int8_t hi);
#endif
