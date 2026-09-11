Implement ONLY this function (Hexagon HVX C):
    void candidate_kernel(const int8_t *A, const int8_t *B,
                          const int32_t *bias, int32_t *C,
                          int M, int N, int K);
Compute row-major int8 matrix multiply with per-column int32 bias:
    C[i*N+j] = sum_k A[i*K+k]*B[k*N+j] + bias[j]
A is [M x K] (row-major), B is [K x N] (row-major), bias is [N] int32 per-column,
C is [M x N] int32. Accumulate inner products in int32, then add bias[j] to each column.
K is not a multiple of 128 (K=130) — handle the K-reduction tail.
Prefer HVX vrmpy for the K-reduction inner loop.
Include <hexagon_types.h> and <hexagon_protos.h>. Do NOT write main().
Respond with a single complete C code block and CLOSE the fence with ```.
