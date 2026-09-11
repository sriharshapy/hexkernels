#ifndef KERNEL_API_H
#define KERNEL_API_H
#include <stdint.h>
/* Dequantize int8 -> int16 (fixed-point scale):
 *   v = ((int32_t)a[i] - (int32_t)zp) * scale
 *   out[i] = clamp( v >> shift, -32768, 32767 )     (arithmetic right shift, toward -inf)
 * Zero-point `zp` (int8) is subtracted first. `scale` is a positive int32
 * multiplier; `shift` >= 0. mult/scale/zp/shift are runtime params -- do NOT
 * hardcode them. Inputs are bounded so (a-zp)*scale fits int16. Large-N /
 * bandwidth-bound variant: the achievability bar streams DDR<->VTCM via uDMA
 * double-buffering. */
void candidate_kernel(const int8_t *a, int16_t *out, int n,
                      int8_t zp, int32_t scale, int shift);
#endif
