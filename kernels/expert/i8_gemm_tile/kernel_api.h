#ifndef KERNEL_API_H
#define KERNEL_API_H
#include <stdint.h>
/* Row-major int8 matmul: C[i*N+j] = sum_k A[i*K+k]*B[k*N+j], int32 accumulate. */
void candidate_kernel(const int8_t *A, const int8_t *B, int32_t *C, int M, int N, int K);
#endif
