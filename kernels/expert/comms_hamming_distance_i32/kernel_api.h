#ifndef KERNEL_API_H
#define KERNEL_API_H
#include <stdint.h>
/* Bitwise Hamming distance per int32 word pair: out[i] = popcount( a[i] ^ b[i] ).
   Result range 0..32. ALL 32 bits counted. n=1024. */
void candidate_kernel(const int32_t *a, const int32_t *b, int32_t *out, int n);
#endif
