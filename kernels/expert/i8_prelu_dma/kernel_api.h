#ifndef KERNEL_API_H
#define KERNEL_API_H
#include <stdint.h>
/* int8 PReLU (per-channel leaky ReLU, fixed-point), large-N bandwidth-bound.
 * Data layout: x[c*n_elem + i], n_ch channels, n_elem elements per channel
 * (n_elem is a multiple of 128). For each channel c and element i:
 *   out[c*n_elem+i] = x[..] > 0 ? x[..]
 *                     : (int8_t)clamp((x[..] * alpha[c]) >> shift, -128, 127)
 * alpha[c] is the per-channel slope numerator (arithmetic right shift, floor).
 * Large-N / bandwidth-bound variant: the achievability bar streams DDR<->VTCM
 * via uDMA double-buffering (tiles are channel-aligned so alpha is per-tile). */
void candidate_kernel(const int8_t *x, int8_t *out,
                      int n_ch, int n_elem,
                      const int8_t *alpha, int shift);
#endif
