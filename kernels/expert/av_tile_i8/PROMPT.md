Implement ONLY this function (Hexagon HVX C):
    void candidate_kernel(const int8_t *A, const int8_t *V, int32_t *O,
                          int M, int N, int D);

Compute attention output tile O = A · V (column-major/pre-transposed V).
A: [M x N] int8 attention scores, row-major (A[i,j] = A[i*N+j]).
V: [D x N] int8 value matrix, COLUMN-major -- V[d,j] = V[d*N+j] (NOT the
   usual [N x D] row-major layout; V is pre-transposed so both A and V are
   contiguous over the reduction axis j).
O: [M x D] int32 output, row-major.

Pinned formula (NO floating point):
  O[i*D + d] = sum_j  A[i*N + j] * V[d*N + j]     (int32 accumulate, j=0..N-1)

M=16, N=16, D=90. D is NOT a multiple of 32 -- handle the output tail.
Output is int32 (no requantization in this kernel).
Prefer HVX vrmpy; include <hexagon_types.h> and <hexagon_protos.h>.
Do NOT write main().
Respond with a single complete C code block and CLOSE the fence with ```.
