#ifndef KERNEL_API_H
#define KERNEL_API_H
#include <stdint.h>
/* int8 hard-swish (fixed-point, truncating division):
 *   relu6_val = clamp(x[i] + 3, 0, 6)
 *   out[i]    = (int8_t)clamp((x[i] * relu6_val) / 6, -128, 127)
 * Division is C integer division (truncates toward zero). No runtime parameters.
 * Large-N / bandwidth-bound variant: the achievability bar streams DDR<->VTCM via
 * uDMA double-buffering. */
void candidate_kernel(const int8_t *x, int8_t *out, int n);
#endif
