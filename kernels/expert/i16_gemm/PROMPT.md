Implement ONLY this function (Hexagon HVX C):
    void candidate_kernel(const int16_t *A, const int16_t *B, int32_t *C,
                          int M, int N, int K);
Compute row-major int16 matmul with int32 accumulator:
    C[i*N+j] = sum_k A[i*K+k] * B[k*N+j]
A is [M x K] = [32 x 130] int16, B is [K x N] = [130 x 32] int16, C is [M x N] int32.
Inputs are bounded to |x| <= 127 so int32 accumulation cannot overflow.
K=130 (NOT a multiple of 128) — handle the K-reduction tail.
Prefer HVX vdmpy (int16 dot-product) for the K-reduction.
Include <hexagon_types.h> and <hexagon_protos.h>. Do NOT write main().
Respond with a single complete C code block and CLOSE the fence with ```.
