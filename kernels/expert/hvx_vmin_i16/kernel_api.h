#ifndef KERNEL_API_H
#define KERNEL_API_H
#include <stdint.h>
/*
 * Elementwise int16 min, n=700 (tail path: 700 = 10*64 + 60).
 * Semantics: out[i] = (a[i] < b[i]) ? a[i] : b[i]  for i in [0, n), using
 * SIGNED int16 comparison (matches Q6_Vh_vmin_VhVh).
 */
void candidate_kernel(const int16_t *a, const int16_t *b, int16_t *out, int n);
#endif
