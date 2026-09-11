#ifndef KERNEL_API_H
#define KERNEL_API_H
#include <stdint.h>
/* Matrix-vector multiply: y[i] = sum_k A[i*K+k]*x[k]
 * A is [M x K] row-major int8, x is [K] int8, y is [M] int32.
 * Equivalent to a fully-connected layer with batch=1. */
void candidate_kernel(const int8_t *A, const int8_t *x, int32_t *y,
                      int M, int K);
#endif
