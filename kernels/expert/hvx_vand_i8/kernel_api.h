#ifndef KERNEL_API_H
#define KERNEL_API_H
#include <stdint.h>
/*
 * Elementwise bitwise AND of two int8 vectors, n=1029 (tail path: 1029 =
 * 8*128 + 5).
 * Semantics: out[i] = (int8_t)((uint8_t)a[i] & (uint8_t)b[i])  for i in [0, n).
 * Pure bitwise op, sign-agnostic (matches Q6_V_vand_VV, the generic
 * bit-pattern AND used regardless of dtype interpretation).
 */
void candidate_kernel(const int8_t *a, const int8_t *b, int8_t *out, int n);
#endif
