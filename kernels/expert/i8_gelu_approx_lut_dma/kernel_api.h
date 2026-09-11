#ifndef KERNEL_API_H
#define KERNEL_API_H
#include <stdint.h>
/* GELU approximation via 256-entry int8->int8 LUT with input requantisation:
 *   idx = clamp( (int)in[i] * scale / 64 + 128, 0, 255 )     ( / is C trunc toward 0 )
 *   out[i] = lut[idx]
 * `scale` is an int8 requant scale in [1,127]; `lut` is a 256-entry runtime table.
 * scale and lut are runtime parameters -- do NOT hardcode them. Large-N /
 * bandwidth-bound variant: the achievability bar streams the big input DDR<->VTCM
 * via uDMA double-buffering (the 256-byte table stays resident on-chip). */
void candidate_kernel(const int8_t *in, int8_t *out, int n,
                      const int8_t *lut, int8_t scale);
#endif
