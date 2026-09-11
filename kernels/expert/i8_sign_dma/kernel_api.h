#ifndef KERNEL_API_H
#define KERNEL_API_H
#include <stdint.h>
/* Signum: out[i] = (in[i] > 0) ? 1 : (in[i] < 0) ? -1 : 0.
 * in/out are signed int8; output values in {-1,0,1}.
 * Large-N / bandwidth-bound variant: the achievability bar streams DDR<->VTCM
 * via uDMA double-buffering. */
void candidate_kernel(const int8_t *in, int8_t *out, int n);
#endif
