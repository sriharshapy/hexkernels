Implement ONLY this function (Hexagon HVX C):
    #define BATCH 4
    void candidate_kernel(const int8_t *A, const int8_t *B,
                          const int32_t *bias, int32_t *C,
                          int M, int N, int K);
Batched int8 matmul + bias -> int32:
  For each batch b in [0, BATCH):
    C[b*M*N + i*N + j] = sum_k A[b*M*K + i*K + k] * B[b*K*N + k*N + j] + bias[b*N + j]
A is [BATCH x M x K] int8 row-major. B is [BATCH x K x N] int8 row-major.
bias is [BATCH x N] int32 -- one bias per (batch, column); index as bias[b*N + j].
C is [BATCH x M x N] int32.
BATCH=4, M=16, N=16, K=32.
Each batch is independent. The bias is per-batch and per-column (broadcasts across rows).
Prefer HVX vrmpy for the K-reduction. Include <hexagon_types.h> and <hexagon_protos.h>. Do NOT write main().
Respond with a single complete C code block and CLOSE the fence with ```.
