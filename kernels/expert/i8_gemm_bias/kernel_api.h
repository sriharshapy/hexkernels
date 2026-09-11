#ifndef KERNEL_API_H
#define KERNEL_API_H
#include <stdint.h>
/* Row-major int8 matmul with per-column int32 bias:
 *   C[i*N+j] = sum_k A[i*K+k]*B[k*N+j] + bias[j]
 * A is [M x K], B is [K x N], bias is [N], C is [M x N] int32.
 * Accumulate inner products in int32 before adding bias. */
void candidate_kernel(const int8_t *A, const int8_t *B,
                      const int32_t *bias, int32_t *C,
                      int M, int N, int K);
#endif
