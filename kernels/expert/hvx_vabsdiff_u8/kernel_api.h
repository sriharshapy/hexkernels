#ifndef KERNEL_API_H
#define KERNEL_API_H
#include <stdint.h>
/*
 * Elementwise unsigned absolute difference, uint8, n=1069 (tail path:
 * 1069 = 8*128 + 45).
 * Semantics: out[i] = (uint8_t)abs((int)a[i] - (int)b[i])  for i in
 * [0, n), computed in wider (int) precision (matches HVX
 * Q6_Vub_vabsdiff_VubVub). Result is always in [0,255], no saturation
 * needed (max |255-0|=255).
 */
void candidate_kernel(const uint8_t *a, const uint8_t *b, uint8_t *out, int n);
#endif
