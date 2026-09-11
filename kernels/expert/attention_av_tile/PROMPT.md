Implement ONLY this function (Hexagon HVX C):
 void candidate_kernel(const int8_t *A, const int8_t *V, int32_t *O,
 int M, int N, int D);

Compute attention output tile O = A · V.
A: [M x N] int8 attention scores, row-major.
V: [N x D] int8 value matrix, row-major.
O: [M x D] int32 output, row-major.

Pinned formula (NO floating point):
 O[i*D + d] = sum_j A[i*N + j] * V[j*D + d] (int32 accumulate, j=0..N-1)

M=16, N=16, D=130. D is NOT a multiple of 128 — handle the tail.
Output is int32 (no requantization in this kernel).
Do NOT write main.
Respond with a single complete C code block and CLOSE the fence with ```.
