#ifndef KERNEL_API_H
#define KERNEL_API_H
#include <stdint.h>
/*
 * Deinterleave: split interleaved in[2*n] into a[n] and b[n].
 *   a[i] = in[2*i]
 *   b[i] = in[2*i+1]
 * n=500. Pure data movement, no arithmetic.
 */
void candidate_kernel(const int8_t *in, int8_t *a, int8_t *b, int n);
#endif
