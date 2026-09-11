#ifndef KERNEL_API_H
#define KERNEL_API_H
#include <stdint.h>
/* 3-tap FIR conv1d, edge-replicated boundaries, bias+shift requant: i8->i8.
 * out[i] = sat_i8( round_half_away_from_zero(
 *              w[0]*a[clamp(i-1,0,n-1)] + w[1]*a[i] + w[2]*a[clamp(i+1,0,n-1)]
 *              + bias, shift) )
 * using the sign-aware form (mult is implicitly 1 -- taps already scale the
 * sum, so no separate multiply is needed):
 *   t    = w[0]*left + w[1]*center + w[2]*right + bias   (int32)
 *   sm   = t >> 31 (arithmetic)
 *   abs  = (t ^ sm) - sm
 *   half = shift > 0 ? (1 << (shift-1)) : 0
 *   sh   = (abs + half) >> shift (arithmetic)
 *   r    = (sh ^ sm) - sm
 *   out[i] = clamp(r, -128, 127)
 * w[3] (int8), bias (int32), shift (int) are runtime params. Border is
 * clamp-to-edge (replicate a[0]/a[n-1]), NOT zero-padding. */
void candidate_kernel(const int8_t *a, const int8_t *w, int32_t bias, int shift,
                      int8_t *out, int n);
#endif
