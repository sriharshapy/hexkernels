#ifndef KERNEL_API_H
#define KERNEL_API_H
#include <stdint.h>
/*
 * 4-wide int8 reduce-add to int32, g=106 groups (tail path: 106 groups
 * = 3*32 + 10; input length n = 4*g = 424 int8 elements).
 * Semantics: for k in [0, g):
 *   out[k] = sum_{j=0..3} (int32)a[4*k+j]
 * SIGNED int8 accumulation (matches Q6_Vw_vrmpy_VbVb(a, ones) where
 * ones is a vector of all int8 value 1 -- a dot product against 1s is
 * exactly a sum). No saturation needed: |sum| <= 4*128 fits easily in
 * int32.
 */
void candidate_kernel(const int8_t *a, int32_t *out, int g);
#endif
