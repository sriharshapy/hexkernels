#ifndef KERNEL_API_H
#define KERNEL_API_H
#include <stdint.h>
/* int8 saturating add of scalar bias: out[i] = (int8_t)clamp(x[i] + bias, -128, 127).
 * bias is a runtime int32 scalar (in this variant it lies within int8 range).
 * Large-N / bandwidth-bound variant: the achievability bar streams DDR<->VTCM
 * via uDMA double-buffering. Do NOT hardcode bias. */
void candidate_kernel(const int8_t *x, int8_t *out, int n, int32_t bias);
#endif
