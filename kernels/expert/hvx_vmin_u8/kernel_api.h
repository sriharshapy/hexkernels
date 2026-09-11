#ifndef KERNEL_API_H
#define KERNEL_API_H
#include <stdint.h>
/*
 * Elementwise unsigned min, uint8, n=947 (tail path: 947 = 7*128 + 51).
 * Semantics: out[i] = (a[i] < b[i]) ? a[i] : b[i]  for i in [0, n), using
 * UNSIGNED comparison (matches HVX Q6_Vub_vmin_VubVub -- NOT the signed
 * Q6_Vb_vmin_VbVb). For example min(128,127)=127 with unsigned compare
 * (a signed compare would say 128, since 128 as int8 is -128, the
 * smallest possible signed byte).
 */
void candidate_kernel(const uint8_t *a, const uint8_t *b, uint8_t *out, int n);
#endif
