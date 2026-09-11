#ifndef KERNEL_API_H
#define KERNEL_API_H
#include <stdint.h>
/*
 * Logical shift-right, uint16, n=619 (tail path: 619 = 9*64 + 43), fixed
 * shift amount.
 * Semantics: out[i] = (uint16_t)(a[i] >> shift)  for i in [0, n).
 * Zero-fill from the top (matches HVX Q6_Vuh_vlsr_VuhR -- unsigned/logical
 * shift, NOT arithmetic; no sign extension since the dtype is unsigned).
 */
void candidate_kernel(const uint16_t *a, uint16_t *out, int n, int shift);
#endif
