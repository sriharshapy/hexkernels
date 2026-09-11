#ifndef KERNEL_API_H
#define KERNEL_API_H
#include <stdint.h>
/* Rescale int16 -> int8, bandwidth-bound large-N variant.
 * Requant params are BAKED as fixed constants (kept out of the signature):
 *   MULT = 3, SHIFT = 4, ZP = 0
 * Semantics (bit-exact vs the int32 scalar golden):
 *   v    = (int32)a[i] * MULT
 *   half = (SHIFT > 0) ? (1 << (SHIFT-1)) : 0
 *   r    = (v >= 0) ? (v + half) >> SHIFT : -(((-v) + half) >> SHIFT)   (round half away from 0)
 *   r   += ZP
 *   out[i] = sat8(r)                                   (clamp to [-128,127])
 * Achievability bar streams a[] DDR<->VTCM via uDMA double-buffering. */
void candidate_kernel(const int16_t *a, int8_t *out, int n);
#endif
