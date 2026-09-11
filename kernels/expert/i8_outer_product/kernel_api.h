#ifndef KERNEL_API_H
#define KERNEL_API_H
#include <stdint.h>
/* Outer product: C[i*N+j] = a[i] * b[j]
 * a is [M] int8, b is [N] int8, C is [M x N] int32. */
void candidate_kernel(const int8_t *a, const int8_t *b, int32_t *C,
                      int M, int N);
#endif
