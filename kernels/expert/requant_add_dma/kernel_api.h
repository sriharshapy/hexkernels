#ifndef KERNEL_API_H
#define KERNEL_API_H
#include <stdint.h>
/* Residual-add requantize (int32 + int32 -> int8), bandwidth-bound large-N variant.
 * Requant params are BAKED as fixed constants (kept out of the signature so the
 * contract is the clean (a, b, out, n) shape):
 *   MULT = 5, SHIFT = 3, ZP = 0
 * Semantics (bit-exact vs the int64 scalar golden):
 *   sum  = (int64)a[i] + (int64)b[i]
 *   v    = sum * MULT
 *   half = (SHIFT > 0) ? (1 << (SHIFT-1)) : 0
 *   r    = (v >= 0) ? (v + half) >> SHIFT : -(((-v) + half) >> SHIFT)   (round half away from 0)
 *   r   += ZP
 *   out[i] = sat8(r)                                   (clamp to [-128,127])
 * Achievability bar streams both inputs DDR<->VTCM via uDMA double-buffering. */
void candidate_kernel(const int32_t *a, const int32_t *b, int8_t *out, int n);
#endif
