#ifndef KERNEL_API_H
#define KERNEL_API_H
#include <stdint.h>
/* int8 ReLU6: out[i] = clamp(x[i], 0, cap) where cap = 6 * scale.
 * scale is a runtime int8 parameter (1 <= scale <= 21 so cap fits in int8).
 * out[i] = x[i] < 0 ? 0 : (x[i] > cap ? cap : x[i]).
 * Large-N / bandwidth-bound variant: the achievability bar streams DDR<->VTCM
 * via uDMA double-buffering. */
void candidate_kernel(const int8_t *x, int8_t *out, int n, int8_t scale);
#endif
