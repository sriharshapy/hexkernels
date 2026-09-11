#ifndef KERNEL_API_H
#define KERNEL_API_H
#include <stdint.h>
/*
 * Elementwise uint8 bitwise OR, n=1033 (tail path: 1033 = 8*128 + 9).
 * Semantics: out[i] = (uint8_t)(a[i] | b[i]) for i in [0, n).
 * Matches Q6_V_vor_VV (raw bit-pattern OR, dtype-agnostic).
 */
void candidate_kernel(const uint8_t *a, const uint8_t *b, uint8_t *out, int n);
#endif
