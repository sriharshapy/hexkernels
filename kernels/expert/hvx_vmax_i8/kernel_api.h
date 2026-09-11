#ifndef KERNEL_API_H
#define KERNEL_API_H
#include <stdint.h>
/*
 * Elementwise int8 max, n=1050 (tail path: 1050 = 8*128 + 26).
 * Semantics: out[i] = (a[i] > b[i]) ? a[i] : b[i]  for i in [0, n), using
 * SIGNED int8 comparison (matches Q6_Vb_vmax_VbVb).
 */
void candidate_kernel(const int8_t *a, const int8_t *b, int8_t *out, int n);
#endif
