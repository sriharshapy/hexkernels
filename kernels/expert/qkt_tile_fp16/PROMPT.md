Implement ONLY this function (Hexagon HVX C):
    void candidate_kernel(const __fp16 *Q, const __fp16 *K, __fp16 *S,
                          int M, int N, int D, _Float16 scale);

(`scale` is typed `_Float16`, not `__fp16`, only because this hexagon-clang
target cannot pass `__fp16` by value as a function argument -- `_Float16` is
the identical IEEE-754 binary16 format and is still a genuine runtime fp16
scalar.)

Compute scaled attention QKᵀ tile in fp16: S = scale * (Q · Kᵀ).
Q: [M x D] fp16, row-major (Q[i,d] = Q[i*D+d]).
K: [D x N] fp16, COLUMN-major per key (K[d,j] = K[d*N+j]) -- NOT row-major;
   K is stored transposed relative to the usual [N x D] layout.
S: [M x N] fp16 output.

Pinned formula (float32-accumulate, two explicit fp16 roundings):
  1. acc[i,j] = sum_d (float)Q[i*D+d] * (float)K[d*N+j]     (d = 0..D-1)
  2. m        = (__fp16)acc[i,j]                             (round to fp16)
  3. S[i,j]   = (__fp16)((float)m * (float)scale)            (round to fp16 again)

M=8, N=8, D=40 (D=40 is less than the 64 fp16-lanes-per-HVX-vector -- zero-pad
the reduction tail; do not read out of bounds).
`scale` is a RUNTIME fp16 parameter -- it must be read at runtime, not
hardcoded (multiple distinct scale values are swept by the harness).
Output is compared to a float32 scalar reference with an fp16 tolerance --
HVX float arithmetic is non-IEEE (qf16), so bit-exactness is NOT required.

Prefer HVX qf16 vector ops for the D-reduction; include <hexagon_types.h> and
<hexagon_protos.h>. Do NOT write main().
Respond with a single complete C code block and CLOSE the fence with ```.
