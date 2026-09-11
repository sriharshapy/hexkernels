#ifndef KERNEL_API_H
#define KERNEL_API_H
#include <stdint.h>
/* int8 ReLU: out[i] = (x[i] < 0) ? 0 : x[i].  Large-N / bandwidth-bound:
 * the achievability bar streams DDR<->VTCM via a single-buffer uDMA staging. */
void candidate_kernel(const int8_t *x, int8_t *out, int n);
#endif
