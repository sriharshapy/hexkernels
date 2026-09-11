#ifndef KERNEL_API_H
#define KERNEL_API_H
#include <stdint.h>
/*
 * Elementwise int8 bitwise NOT (ones complement), n=1017
 * (tail path: 1017 = 7*128 + 121).
 * Semantics: out[i] = (int8_t)(~a[i]) for i in [0, n).
 * Matches Q6_V_vnot_V (raw bit-pattern complement).
 */
void candidate_kernel(const int8_t *a, int8_t *out, int n);
#endif
