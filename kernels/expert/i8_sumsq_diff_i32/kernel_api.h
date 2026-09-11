#ifndef KERNEL_API_H
#define KERNEL_API_H
#include <stdint.h>
/* Sum of squared differences (L2-distance-squared):
   out[0] = sum_i (a[i] - b[i])^2, accumulated in int32.
   Inputs are signed int8. Result fits in int32 for n=1024, max diff=255. */
void candidate_kernel(const int8_t *a, const int8_t *b, int n, int32_t *out);
#endif
