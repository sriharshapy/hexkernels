#ifndef KERNEL_API_H
#define KERNEL_API_H
#include <stdint.h>
/*
 * Negative average, int8, n=1091 (tail path: 1091 = 8*128 + 67).
 * Semantics: out[i] = (int8_t)(((int)a[i] - (int)b[i]) >> 1)  for i in
 * [0, n). The shift is an ARITHMETIC (floor) shift, computed in wider
 * (int) precision then narrowed -- NO rounding bias and NO saturation
 * (matches HVX Q6_Vb_vnavg_VbVb). "Negative average" = (a-b)/2 floored,
 * NOT -(a+b)/2 -- verify against the pinned edge cases below.
 */
void candidate_kernel(const int8_t *a, const int8_t *b, int8_t *out, int n);
#endif
