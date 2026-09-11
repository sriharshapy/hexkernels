#ifndef KERNEL_API_H
#define KERNEL_API_H
#include <stdint.h>
/*
 * Sign-extending unpack: int8 -> int16, n=1000 (tail path: 1000 = 7*128 + 104).
 * Semantics: out[i] = (int16_t)(int8_t)a[i]  for i in [0, n).
 * Matches Q6_Wh_vunpack_Vb (sign-extends each byte; NOT zero-extend).
 */
void candidate_kernel(const int8_t *a, int16_t *out, int n);
#endif
