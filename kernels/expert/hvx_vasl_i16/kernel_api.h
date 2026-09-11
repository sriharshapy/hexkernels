#ifndef KERNEL_API_H
#define KERNEL_API_H
#include <stdint.h>
/*
 * Arithmetic shift-left, int16, n=583 (tail path: 583 = 9*64 + 7), fixed
 * shift amount.
 * Semantics: out[i] = (int16_t)((uint16_t)a[i] << shift)  for i in [0, n).
 * Two's-complement WRAP on bits shifted past bit 15 (matches HVX
 * Q6_Vh_vasl_VhR -- NOT a saturating shift; bits simply fall off the top).
 */
void candidate_kernel(const int16_t *a, int16_t *out, int n, int shift);
#endif
