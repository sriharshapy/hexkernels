#ifndef KERNEL_API_H
#define KERNEL_API_H
#include <stdint.h>
/* Requantize int32 -> int8 with fused ReLU, bandwidth-bound large-N variant.
 * Requant params are BAKED as fixed constants (kept out of the signature):
 *   MULT = 13, SHIFT = 3, ZP = 0
 * Semantics (bit-exact vs the int64 scalar golden):
 *   v    = (int64)a[i] * MULT
 *   half = (SHIFT > 0) ? (1 << (SHIFT-1)) : 0
 *   r    = (v >= 0) ? (v + half) >> SHIFT : -(((-v) + half) >> SHIFT)   (round half away from 0)
 *   r   += ZP
 *   r    = max(r, ZP)                                  (fused ReLU floor at zp)
 *   out[i] = sat8(r)                                   (clamp to [-128,127])
 * Achievability bar streams a[] DDR<->VTCM via uDMA double-buffering. */
void candidate_kernel(const int32_t *a, int8_t *out, int n);
#endif
