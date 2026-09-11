Implement ONLY this function (Hexagon HVX C):
    void candidate_kernel(const int8_t *Q, const int8_t *K, int8_t *S,
                          int M, int N, int D,
                          int32_t scale_mult, int scale_shift);

Compute scaled attention QKᵀ tile: S = requantize(Q · Kᵀ).
Q: [M x D] int8, row-major (Q[i,d] = Q[i*D+d]).
K: [D x N] int8, COLUMN-major per key (K[d,j] = K[d*N+j]) -- NOT row-major;
   K is stored transposed relative to the usual [N x D] layout.
S: [M x N] int8 output.

Pinned multi-step integer formula (NO floating point):
  1. raw[i,j] = sum_d  Q[i*D+d] * K[d*N+j]               (int32 accumulate over d=0..D-1)
  2. Requantize raw[i,j] -> int8:
       r    = (int64_t)raw[i,j] * (int64_t)scale_mult
       half = (scale_shift > 0) ? (1LL << (scale_shift-1)) : 0
       q    = (r >= 0) ? ((r + half) >> scale_shift)
                       : -((-r + half) >> scale_shift)       (round-half-away-from-zero)
       S[i*N+j] = clamp(q, -128, 127)

M=16, N=16, D=98 (D is NOT a multiple of 128 -- handle the reduction tail
when accumulating in groups of 4 bytes for vrmpy).
scale_mult and scale_shift are runtime parameters (anti-hardcode -- multiple
param sets are used, including a scale_shift=0 case with no rounding).
Prefer HVX vrmpy for the D-reduction; include <hexagon_types.h> and <hexagon_protos.h>.
Do NOT use floating point. Do NOT write main().
Respond with a single complete C code block and CLOSE the fence with ```.
