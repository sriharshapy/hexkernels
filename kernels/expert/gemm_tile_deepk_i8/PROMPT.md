Implement ONLY this function (Hexagon HVX C):
    void candidate_kernel(const int8_t *A, const int8_t *B, int32_t *C,
                          int M, int N, int K);
Compute a deep-K int8 GEMM tile with B in the CONVENTIONAL (non-transposed)
[K x N] row-major layout:
    C[i*N+j] = sum over k of A[i*K+k] * B[k*N+j]
A is [M x K] row-major (row i contiguous over k). B is [K x N] row-major --
row k of B is contiguous over j (this is the ordinary GEMM operand layout,
NOT transposed). Accumulate in int32 (no saturation needed at these sizes).
Pinned shapes: M=8, N=8, K=262 -- K is deep and NOT a multiple of 4, so a
vrmpy-style 4-byte-group reduction has a genuine tail group of only 2 valid
bytes after 65 full groups; handle it correctly (zero-pad, don't drop it).
Prefer HVX vrmpy for the K-reduction. Include <hexagon_types.h> and
<hexagon_protos.h>. Do NOT write main().
Respond with a single complete C code block and CLOSE the fence with ```.
