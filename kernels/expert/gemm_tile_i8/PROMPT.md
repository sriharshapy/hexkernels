Implement ONLY this function (Hexagon HVX C):
    void candidate_kernel(const int8_t *A, const int8_t *B, int32_t *C, int M, int N, int K);
Compute an int8 GEMM tile where A is [M x K] row-major (row i contiguous over k) and
B is [N x K] row-major -- i.e. B is given ALREADY TRANSPOSED relative to the conventional
[K x N] GEMM operand, so row j of B is contiguous over k, exactly like row i of A:
    C[i*N+j] = sum over k of A[i*K+k]*B[j*K+k]
Accumulate in int32 (no saturation needed at these sizes). Pinned shapes: M=24, N=24,
K=100 -- K is not a multiple of 128, so handle the reduction tail. Prefer HVX vrmpy for
the K-reduction. Include <hexagon_types.h> and <hexagon_protos.h>. Do NOT write main().
Respond with a single complete C code block and CLOSE the fence with ```.
