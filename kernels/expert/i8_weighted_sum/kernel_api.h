#ifndef KERNEL_API_H
#define KERNEL_API_H
#include <stdint.h>
/* Weighted sum: out[0] = sum_i w[i] * a[i], int32 accumulator.
   Both a and w are signed int8. Semantically identical to dot product
   but framed as data-with-weights (a=data, w=weights). */
void candidate_kernel(const int8_t *a, const int8_t *w, int n, int32_t *out);
#endif
