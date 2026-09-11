#ifndef KERNEL_API_H
#define KERNEL_API_H
#include <stdint.h>
/* Residual-add requantize (skip-connection path):
 *   sum  = (int64_t)a[i] + (int64_t)b[i]   (widened to avoid int32 overflow)
 *   v    = sum * (int64_t)mult
 *   half = shift > 0 ? (1LL << (shift-1)) : 0
 *   r    = (v >= 0) ? (v + half) >> shift : -(((-v) + half) >> shift)
 *   r   += zp
 *   out[i] = saturate_to_int8(r)             (clamp to [-128, 127])
 * mult, shift, zp are runtime parameters -- do NOT hardcode them. */
void candidate_kernel(const int32_t *a, const int32_t *b, int8_t *out, int n,
                      int32_t mult, int shift, int8_t zp);
#endif
