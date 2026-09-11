#ifndef KERNEL_API_H
#define KERNEL_API_H
#include <stdint.h>
/* Element-wise max with scalar: out[i] = max(in[i], c) (signed int8; c signed int8).
 * c=0 is ReLU. Large-N / bandwidth-bound variant: the achievability bar streams
 * DDR<->VTCM via uDMA double-buffering. c is a runtime scalar (do NOT hardcode). */
void candidate_kernel(const int8_t *in, int8_t *out, int n, int8_t c);
#endif
