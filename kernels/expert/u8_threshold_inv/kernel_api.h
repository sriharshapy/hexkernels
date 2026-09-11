#ifndef KERNEL_API_H
#define KERNEL_API_H
#include <stdint.h>
/* Inverse threshold: out[i] = (in[i] < thresh) ? 255 : 0.
   Comparison is STRICT less-than: at in[i]==thresh the output is 0. */
void candidate_kernel(const uint8_t *in, uint8_t *out, int n, uint8_t thresh);
#endif
