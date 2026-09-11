Implement ONLY this function (Hexagon HVX C):
    void candidate_kernel(const int8_t *A, const int8_t *x, int32_t *y,
                          int M, int K);
Compute int8 matrix-vector multiply (fully-connected layer, batch=1):
    y[i] = sum_k A[i*K+k] * x[k]   for i in [0, M)
A is [M x K] row-major int8, x is [K] int8, y is [M] int32. Accumulate in int32.
M=256, K=130 (NOT a multiple of 128) — handle the K-reduction tail.
Prefer HVX vrmpy for the K-reduction inner loop over each row.
Include <hexagon_types.h> and <hexagon_protos.h>. Do NOT write main().
Respond with a single complete C code block and CLOSE the fence with ```.
