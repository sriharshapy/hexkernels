#ifndef KERNEL_API_H
#define KERNEL_API_H
#include <stdint.h>
/* Requantize int32 -> int8, DMA double-buffered over N tiles:
 *   out[i] = sat_i8( round_half_away_from_zero(a[i]*mult >> shift) + zp )
 * with the sign-aware form (matches the sign-magnitude requant used
 * elsewhere in this benchmark):
 *   sm  = a[i] >> 31 (arithmetic)          -- 0 if a[i]>=0, -1 if a[i]<0
 *   abs = (a[i] ^ sm) - sm                 -- |a[i]|
 *   am  = abs * mult
 *   half = shift>0 ? (1<<(shift-1)) : 0
 *   sh  = (am + half) >> shift (arithmetic)
 *   r   = (sh ^ sm) - sm + zp
 *   out[i] = clamp(r, -128, 127)
 * mult, shift, zp are runtime params (harness fixes mult=200, shift=8, zp=5).
 * Large-N / bandwidth-bound; the achievability bar DMA-tiles a[] into VTCM,
 * requantizes on-chip, and DMAs the int8 result back out, double-buffered. */
void candidate_kernel(const int32_t *a, int8_t *out, int n, int32_t mult, int shift, int8_t zp);
#endif
