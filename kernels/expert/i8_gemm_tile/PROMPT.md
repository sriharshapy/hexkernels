Implement ONLY this function (Hexagon HVX C):
    void candidate_kernel(const int8_t *A, const int8_t *B, int32_t *C, int M, int N, int K);
Compute row-major int8 matrix multiply C = A*B where A is [M x K], B is [K x N],
C is [M x N] int32: C[i*N+j] = sum over k of A[i*K+k]*B[k*N+j]. Accumulate in int32.
K is not a multiple of 128 — handle the reduction tail. Prefer HVX vrmpy for the K-reduction.
Include <hexagon_types.h> and <hexagon_protos.h>. Do NOT write main().
Respond with a single complete C code block and CLOSE the fence with ```.
