#ifndef KERNEL_API_H
#define KERNEL_API_H
#include <stdint.h>
/* Batched int8 matmul: for each batch b in [0, BATCH):
 *   C[b*M*N + i*N + j] = sum_k A[b*M*K + i*K + k] * B[b*K*N + k*N + j]
 * A is [BATCH x M x K] int8, B is [BATCH x K x N] int8, C is [BATCH x M x N] int32.
 * All matrices are row-major. Each batch is independent. */
#define BATCH 4
void candidate_kernel(const int8_t *A, const int8_t *B, int32_t *C,
                      int M, int N, int K);
#endif
