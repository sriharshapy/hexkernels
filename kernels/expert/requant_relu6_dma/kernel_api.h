#ifndef KERNEL_API_H
#define KERNEL_API_H
#include <stdint.h>
/* Requantize int32 -> int8 with fused ReLU6, bandwidth-bound large-N variant.
 * Requant params are BAKED as fixed constants (kept out of the signature):
 *   MULT = 13, SHIFT = 3, ZP = 0, Q6 = 24  (quantized value of 6.0 in output scale)
 * Semantics (bit-exact vs the int64 scalar golden):
 *   v    = (int64)a[i] * MULT
 *   half = (SHIFT > 0) ? (1 << (SHIFT-1)) : 0
 *   r    = (v >= 0) ? (v + half) >> SHIFT : -(((-v) + half) >> SHIFT)   (round half away from 0)
 *   r   += ZP
 *   r    = clamp(r, ZP, ZP + Q6)                       (fused ReLU6)
 *   out[i] = sat8(r)                                   (clamp to [-128,127])
 * Achievability bar streams a[] DDR<->VTCM via uDMA double-buffering. */
void candidate_kernel(const int32_t *a, int8_t *out, int n);
#endif
