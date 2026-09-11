#ifndef KERNEL_API_H
#define KERNEL_API_H
#include <stdint.h>
/* Row-major int16 matmul with int32 accumulator:
 *   C[i*N+j] = sum_k A[i*K+k] * B[k*N+j]
 * A is [M x K] int16, B is [K x N] int16, C is [M x N] int32.
 * Inputs bounded to |x| <= 127 by harness so K=130 products fit in int32. */
void candidate_kernel(const int16_t *A, const int16_t *B, int32_t *C,
                      int M, int N, int K);
#endif
