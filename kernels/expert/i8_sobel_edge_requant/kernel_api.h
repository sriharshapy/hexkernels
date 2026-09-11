#ifndef KERNEL_API_H
#define KERNEL_API_H
#include <stdint.h>
/* Sobel gradient magnitude + requantize to uint8.
 * Input: uint8 image in[h*w].
 * For each pixel compute Sobel magnitude: mag = |Gx| + |Gy| (L1 norm, max 1020).
 * Requantize: out[y*w+x] = sat_u8(round(mag * mult >> shift) + zp)
 *   where round is round-half-up (mag >= 0 always):
 *     rounded = (mag * mult + half) >> shift, half = (1 << shift) >> 1 = shift>0 ? (1<<(shift-1)) : 0
 *   then clamp to [0, 255].
 * Border: clamp-to-edge.
 * mult, shift, zp are runtime params -- do NOT hardcode. */
void candidate_kernel(const uint8_t *in, uint8_t *out, int w, int h,
                      int mult, int shift, int zp);
#endif
