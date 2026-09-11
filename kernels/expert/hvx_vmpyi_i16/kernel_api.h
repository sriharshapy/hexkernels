#ifndef KERNEL_API_H
#define KERNEL_API_H
#include <stdint.h>
/*
 * Elementwise int16 multiply (LOW 16 bits of the product), n=577 (tail
 * path: 577 = 9*64 + 1).
 * Semantics: out[i] = (int16_t)((int)a[i] * (int)b[i])  for i in [0, n) --
 * i.e. the low 16 bits of the exact 32-bit product, reinterpreted as
 * signed two's-complement int16 (a plain truncating integer multiply, NOT
 * a Q15 fixed-point fractional multiply). Matches Q6_Vh_vmpyi_VhVh.
 */
void candidate_kernel(const int16_t *a, const int16_t *b, int16_t *out, int n);
#endif
