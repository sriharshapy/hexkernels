Implement ONLY this function (Hexagon HVX C):
    void candidate_kernel(const int8_t *a, const int8_t *b, int32_t *C,
                          int M, int N);
Compute the int8 outer product (rank-1 matrix):
    C[i*N+j] = a[i] * b[j]   for i in [0,M), j in [0,N)
a is [M]=128 int8, b is [N]=128 int8, C is [M x N]=128x128 int32.
For each row i, broadcast a[i] and multiply element-wise with b[0..N-1].
HVX hint: for each i, splat a[i] into a vector and use vmpy against the b vector.
Include <hexagon_types.h> and <hexagon_protos.h>. Do NOT write main().
Respond with a single complete C code block and CLOSE the fence with ```.
