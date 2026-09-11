#ifndef KERNEL_API_H
#define KERNEL_API_H
#include <stdint.h>
/*
 * Elementwise unsigned max, uint8, n=1013 (tail path: 1013 = 7*128 + 117).
 * Semantics: out[i] = (a[i] > b[i]) ? a[i] : b[i]  for i in [0, n), using
 * UNSIGNED comparison (matches HVX Q6_Vub_vmax_VubVub -- NOT the signed
 * Q6_Vb_vmax_VbVb). For example max(128,127)=128 with unsigned compare
 * (a signed compare would say 127, since 128 as int8 is -128).
 */
void candidate_kernel(const uint8_t *a, const uint8_t *b, uint8_t *out, int n);
#endif
