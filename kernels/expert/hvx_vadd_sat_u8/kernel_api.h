#ifndef KERNEL_API_H
#define KERNEL_API_H
#include <stdint.h>
/*
 * Elementwise SATURATING uint8 add, n=777 (tail path: 777 = 6*128 + 9).
 * Semantics: out[i] = sat_u8((int)a[i] + (int)b[i])  for i in [0, n),
 * where sat_u8(x) clamps x to [0, 255]. Matches Q6_Vub_vadd_VubVub_sat.
 */
void candidate_kernel(const uint8_t *a, const uint8_t *b, uint8_t *out, int n);
#endif
