#ifndef KERNEL_API_H
#define KERNEL_API_H
#include <stdint.h>
/*
 * Elementwise int8 add, n=1037 (tail path: 1037 = 8*128 + 13).
 * Semantics: out[i] = (int8_t)((int)a[i] + (int)b[i])  for i in [0, n).
 * Two's-complement WRAP on overflow (matches Q6_Vb_vadd_VbVb -- NOT saturating).
 */
void candidate_kernel(const int8_t *a, const int8_t *b, int8_t *out, int n);
#endif
