#ifndef KERNEL_API_H
#define KERNEL_API_H
#include <stdint.h>
/* int8 leaky ReLU (fixed-point): out[i] = x[i] > 0 ? x[i] : sat8((x[i]*alpha) >> shift).
 * alpha (positive multiplier) and shift (arithmetic right-shift count) are runtime
 * parameters; the negative-slope result is clamped to [-128,127].
 * Large-N / bandwidth-bound variant: the achievability bar streams DDR<->VTCM
 * via uDMA double-buffering. */
void candidate_kernel(const int8_t *x, int8_t *out, int n, int alpha, int shift);
#endif
