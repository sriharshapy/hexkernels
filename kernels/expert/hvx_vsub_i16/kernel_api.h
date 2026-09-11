#ifndef KERNEL_API_H
#define KERNEL_API_H
#include <stdint.h>
/*
 * Elementwise int16 subtract, n=651 (tail path: 651 = 10*64 + 11).
 * Semantics: out[i] = (int16_t)((int)a[i] - (int)b[i])  for i in [0, n).
 * Two's-complement WRAP on overflow (matches Q6_Vh_vsub_VhVh -- NOT saturating).
 */
void candidate_kernel(const int16_t *a, const int16_t *b, int16_t *out, int n);
#endif
