Implement ONLY this function (Hexagon HVX C):
    void candidate_kernel(const int8_t *A, const int8_t *B, const int32_t *bias, int8_t *C,
                          int M, int N, int K);
A is [M x K] row-major (row i contiguous over k). B is [N x K] row-major -- given
ALREADY TRANSPOSED relative to the conventional [K x N] GEMM operand, so row j of B
is contiguous over k, exactly like row i of A. `bias` is an int32 array of length N,
ONE value per output COLUMN j, broadcast across every row i. Compute:
    acc      = (sum over k of A[i*K+k]*B[j*K+k]) + bias[j]     (bias added BEFORE relu)
    relu     = acc > 0 ? acc : 0
    C[i*N+j] = (int8_t) clamp(relu, 0, 127)
Accumulate the dot product in int32. Add bias BEFORE applying relu (not after).
Pinned shapes: M=20, N=20, K=80. N=20 is not a multiple of 32 int32 lanes -- handle
the output tail (the K=80 reduction has no tail, it's an exact multiple of 4). Prefer
HVX vrmpy for the K-reduction. Include <hexagon_types.h> and <hexagon_protos.h>.
Do NOT write main(). Respond with a single complete C code block and CLOSE the fence
with ```.
