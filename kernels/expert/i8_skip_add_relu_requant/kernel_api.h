#ifndef KERNEL_API_H
#define KERNEL_API_H
#include <stdint.h>
/* Fused skip-connection: add two int8 tensors, apply ReLU, requantize to int8.
 *   sum  = (int32_t)a[i] + (int32_t)b[i]   (widened to avoid int8 overflow)
 *   sum  = max(sum, 0)                       (ReLU)
 *   v    = (int64_t)sum * (int64_t)mult
 *   half = shift > 0 ? (1LL << (shift-1)) : 0
 *   r    = (v >= 0) ? (v + half) >> shift : -(((-v) + half) >> shift)
 *   r   += zp
 *   out[i] = saturate_to_int8(r)             // clamp to [-128, 127]
 * mult, shift, zp are runtime parameters -- do NOT hardcode them. */
void candidate_kernel(const int8_t *a, const int8_t *b, int8_t *out, int n,
                      int32_t mult, int shift, int8_t zp);
#endif
