#ifndef KERNEL_API_H
#define KERNEL_API_H
#include <stdint.h>
/* Fused multiply-add requantize (int32 * int32 + const bias -> int8),
 * bandwidth-bound large-N variant. Inherits v4 mul_add_requant, but the per-element
 * bias array c[] is baked to a fixed scalar constant BIAS so the contract keeps the
 * clean 2-input (a, b, out, n) shape. All params are BAKED constants:
 *   BIAS = 50, MULT = 3, SHIFT = 1, ZP = 0
 * Semantics (bit-exact vs the int64 scalar golden):
 *   fma  = (int64)a[i] * (int64)b[i] + BIAS
 *   v    = fma * MULT
 *   half = (SHIFT > 0) ? (1 << (SHIFT-1)) : 0
 *   r    = (v >= 0) ? (v + half) >> SHIFT : -(((-v) + half) >> SHIFT)   (round half away from 0)
 *   r   += ZP
 *   out[i] = sat8(r)                                   (clamp to [-128,127])
 * Inputs a[], b[] are bounded to [-15,15] so all intermediates fit int16.
 * Achievability bar streams both inputs DDR<->VTCM via uDMA double-buffering. */
void candidate_kernel(const int32_t *a, const int32_t *b, int8_t *out, int n);
#endif
