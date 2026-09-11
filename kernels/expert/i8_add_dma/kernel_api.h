#ifndef KERNEL_API_H
#define KERNEL_API_H
#include <stdint.h>
/* int8 saturating add: out[i] = sat_i8(a[i] + b[i]), clamped to [-128,127].
 * Large-N / bandwidth-bound: the achievability bar streams DDR<->VTCM via a
 * single-buffer uDMA staging. */
void candidate_kernel(const int8_t *a, const int8_t *b, int8_t *out, int n);
#endif
