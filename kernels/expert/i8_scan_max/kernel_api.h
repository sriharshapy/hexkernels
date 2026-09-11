#ifndef KERNEL_API_H
#define KERNEL_API_H
#include <stdint.h>
/* Inclusive prefix maximum of a 1D int8 array, output int8.
 * out[i] = max(in[0], in[1], ..., in[i])  for i in [0, n).
 * out[0] = in[0].  out[i] = max(out[i-1], in[i]). */
void candidate_kernel(const int8_t *in, int8_t *out, int n);
#endif
