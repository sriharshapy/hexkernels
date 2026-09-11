#ifndef KERNEL_API_H
#define KERNEL_API_H
#include <stdint.h>
/*
 * 4-wide int8 dot-product accumulate to int32, g=261 groups (tail path:
 * 261 groups = 8*32 + 5; input length n = 4*g = 1044 int8 elements each).
 * Semantics: for k in [0, g):
 *   out[k] = sum_{j=0..3} (int32)a[4*k+j] * (int32)b[4*k+j]
 * SIGNED x SIGNED int8 dot product (matches Q6_Vw_vrmpy_VbVb). No
 * saturation is needed: |sum| <= 4*127*128 fits easily in int32.
 */
void candidate_kernel(const int8_t *a, const int8_t *b, int32_t *out, int g);
#endif
