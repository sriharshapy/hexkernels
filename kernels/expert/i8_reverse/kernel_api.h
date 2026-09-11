#ifndef KERNEL_API_H
#define KERNEL_API_H
#include <stdint.h>
/*
 * Reverse a 1D array: out[i] = in[n-1-i] for i in [0, n).
 * n=501. Pure data movement, no arithmetic.
 */
void candidate_kernel(const int8_t *in, int8_t *out, int n);
#endif
