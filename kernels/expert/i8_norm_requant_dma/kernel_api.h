#ifndef KERNEL_API_H
#define KERNEL_API_H
#include <stdint.h>
/* Per-channel normalize + requantize (int8 -> int8), bandwidth-bound large-N variant.
 * Inherits v4 i8_norm_requant. Layout is [NUM_CH][per_ch] with NUM_CH baked and
 * per_ch = n / NUM_CH. The per-channel norm arrays and requant params are all BAKED
 * constants so the contract keeps the clean (a, out, n) shape:
 *   NUM_CH   = 16
 *   NORM_MULT[c], NORM_SHIFT[c]  (see prompt.md / the kernel source)
 *   MULT = 5, SHIFT = 4, ZP = 0
 * For channel c and element i (idx = c*per_ch + i):
 *   norm = round_half_away( a[idx]*NORM_MULT[c] >> NORM_SHIFT[c] )
 *   v    = norm * MULT
 *   half = (SHIFT>0) ? (1<<(SHIFT-1)) : 0
 *   r    = round_half_away( v >> SHIFT ) + ZP
 *   out[idx] = sat8(r)                                 (clamp to [-128,127])
 * Achievability bar streams a[] DDR<->VTCM via uDMA double-buffering. */
void candidate_kernel(const int8_t *a, int8_t *out, int n);
#endif
