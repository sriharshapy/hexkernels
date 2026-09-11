Implement ONLY this function (Hexagon HVX C):
    void candidate_kernel(const __fp16 *A, const __fp16 *V, __fp16 *O,
                          int M, int N, int D);

Compute attention output tile O = A · V in fp16 (no requantization).
A: [M x N] fp16 attention probs, row-major (A[i,j] = A[i*N+j]).
V: [D x N] fp16 value matrix, COLUMN-major/pre-transposed (V[d,j] = V[d*N+j])
   -- NOT [N x D] row-major. Both A and V are contiguous over the REDUCTION
   axis j.
O: [M x D] fp16 output, row-major.

Pinned formula (float32-accumulate, single fp16-rounding):
  O[i,d] = (__fp16)( sum_j (float)A[i*N+j] * (float)V[d*N+j] )   (j = 0..N-1)

M=8, D=8, N=40 (N is the reduction axis, less than the 64 fp16-lanes-per-HVX
-vector -- zero-pad the reduction tail; do not read out of bounds).
Output is compared to a float32 scalar reference with an fp16 tolerance --
HVX float arithmetic is non-IEEE (qf16), so bit-exactness is NOT required.

Prefer HVX qf16 vector ops for the N-reduction; include <hexagon_types.h> and
<hexagon_protos.h>. Do NOT write main().
Respond with a single complete C code block and CLOSE the fence with ```.
