#ifndef KERNEL_API_H
#define KERNEL_API_H
#include <stdint.h>
/* Element-wise int8 multiply then requantize to int8:
 *   prod = (int32_t)a[i] * (int32_t)b[i]
 *   v    = (int64_t)prod * (int64_t)mult
 *   half = shift > 0 ? (1LL << (shift-1)) : 0
 *   r    = (v >= 0) ? (v + half) >> shift : -(((-v) + half) >> shift)
 *   r   += zp
 *   out[i] = saturate_to_int8(r)
 * mult, shift, zp are runtime parameters (do NOT hardcode). Large-N /
 * bandwidth-bound variant: the achievability bar streams DDR<->VTCM via uDMA
 * double-buffering. */
void candidate_kernel(const int8_t *a, const int8_t *b, int8_t *out, int n,
                      int32_t mult, int shift, int8_t zp);
#endif
