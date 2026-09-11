Implement ONLY this function (Hexagon HVX C):
    void candidate_kernel(const int8_t *A, const int8_t *B, int32_t *C,
                          int M, int N, int K);
Compute batched int8 matmul with BATCH=4 independent matrix multiplications:
    for b in [0, BATCH):
        C[b*M*N + i*N + j] = sum_k A[b*M*K + i*K + k] * B[b*K*N + k*N + j]
A is [BATCH x M x K] = [4 x 16 x 130] int8, B is [BATCH x K x N] = [4 x 130 x 16] int8.
C is [BATCH x M x N] = [4 x 16 x 16] int32. All row-major. Accumulate in int32.
K=130 (NOT a multiple of 128) — handle the K-reduction tail in every batch.
BATCH is the compile-time constant defined in kernel_api.h (#define BATCH 4).
Prefer HVX vrmpy for the K-reduction. Include <hexagon_types.h> and <hexagon_protos.h>.
Do NOT write main().
Respond with a single complete C code block and CLOSE the fence with ```.
