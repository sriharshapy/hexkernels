#ifndef KERNEL_API_H
#define KERNEL_API_H
#include <stdint.h>
/*
 * Elementwise int8 bitwise XOR, n=1021 (tail path: 1021 = 7*128 + 125).
 * Semantics: out[i] = (int8_t)((uint8_t)a[i] ^ (uint8_t)b[i]) for i in [0, n).
 * Raw bit-pattern XOR (sign-agnostic), matches Q6_V_vxor_VV.
 */
void candidate_kernel(const int8_t *a, const int8_t *b, int8_t *out, int n);
#endif
