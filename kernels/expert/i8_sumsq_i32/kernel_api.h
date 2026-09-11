#ifndef KERNEL_API_H
#define KERNEL_API_H
#include <stdint.h>
/* Sum of squares, int32 accumulator: out[0] = sum_i a[i]*a[i]. */
void candidate_kernel(const int8_t *a, int n, int32_t *out);
#endif
