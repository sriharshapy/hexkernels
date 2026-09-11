#ifndef KERNEL_API_H
#define KERNEL_API_H
#include <stdint.h>
/* Elementwise affine + requantize:
 *   out[i] = sat8( round_half_away_from_zero( (a[i]*scale + shift) ) >> s )
 * scale, shift are runtime int32 scalars; s is the right-shift amount (>=0);
 * rounding adds (1<<(s-1)) to the magnitude before shifting (round half away
 * from zero); result clamps to [-128,127]. Do NOT hardcode the params.
 * Large-N / bandwidth-bound variant: the achievability bar streams DDR<->VTCM
 * via uDMA double-buffering. */
void candidate_kernel(const int8_t *a, int8_t *out, int n,
                      int32_t scale, int32_t shift, int s);
#endif
