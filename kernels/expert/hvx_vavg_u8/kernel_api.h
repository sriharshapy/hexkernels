#ifndef KERNEL_API_H
#define KERNEL_API_H
#include <stdint.h>
/*
 * Elementwise ROUNDING average of two uint8 vectors, n=900 (tail path:
 * 900 = 7*128 + 4).
 * Semantics: out[i] = (uint8_t)(((int)a[i] + (int)b[i] + 1) >> 1)  for i in [0, n).
 * Round-half-up (matches Q6_Vub_vavg_VubVub_rnd -- NOT the floor/truncating
 * Q6_Vub_vavg_VubVub, whose formula is ((a+b)>>1) with no +1).
 */
void candidate_kernel(const uint8_t *a, const uint8_t *b, uint8_t *out, int n);
#endif
