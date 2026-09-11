#ifndef KERNEL_API_H
#define KERNEL_API_H
#include <stdint.h>
/* Batched int8 matmul + bias -> int32:
 *   For each batch b in [0, BATCH):
 *     C[b*M*N + i*N + j] = sum_k A[b*M*K + i*K + k] * B[b*K*N + k*N + j] + bias[b*N + j]
 *
 * A is [BATCH x M x K] int8 row-major.
 * B is [BATCH x K x N] int8 row-major.
 * bias is [BATCH x N] int32 -- per batch, per output column.
 * C is [BATCH x M x N] int32.
 * Each batch is independent. bias[b*N+j] is added to every row i in batch b. */
#define BATCH 4
void candidate_kernel(const int8_t *A, const int8_t *B,
                      const int32_t *bias, int32_t *C,
                      int M, int N, int K);
#endif
