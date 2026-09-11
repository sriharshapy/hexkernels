#ifndef KERNEL_API_H
#define KERNEL_API_H
#include <stdint.h>
/*
 * Saturating elementwise int16 add, n=703 (tail path: 703 = 10*64 + 63).
 * Semantics: out[i] = sat16((int)a[i] + (int)b[i])  for i in [0, n),
 * where sat16 clamps to [-32768, 32767]. Matches HVX Q6_Vh_vadd_VhVh_sat
 * (NOT the wrapping Q6_Vh_vadd_VhVh).
 */
void candidate_kernel(const int16_t *a, const int16_t *b, int16_t *out, int n);
#endif
