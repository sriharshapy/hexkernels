Implement ONLY this function (Hexagon HVX C):
 void candidate_kernel(const int8_t *Q, const int8_t *K, int8_t *S,
 int M, int N, int D,
 int32_t scale_mult, int scale_shift);

Compute scaled attention QKᵀ tile: S = requantize(Q · Kᵀ).
Q: [M x D] int8, row-major. K: [N x D] int8, row-major (K is ALREADY row-major, NOT transposed).
S: [M x N] int8 output.

Pinned multi-step integer formula (NO floating point):
 1. raw[i,j] = sum_d Q[i*D+d] * K[j*D+d] (int32 accumulate over d=0..D-1)
 (NOTE: K is indexed K[j*D+d] — you are computing Q * Kᵀ by summing along D)
 2. Requantize raw[i,j] -> int8:
 r = (int64_t)raw[i,j] * (int64_t)scale_mult
 half = (scale_shift > 0) ? (1LL << (scale_shift-1)) : 0
 q = (r >= 0) ? ((r + half) >> scale_shift)
 : -((-r + half) >> scale_shift) (round-half-away-from-zero)
 S[i*N+j] = clamp(q, -128, 127)

M=16, N=16, D=130 (D is NOT a multiple of 128 — handle the reduction tail).
scale_mult and scale_shift are runtime parameters (anti-hardcode — multiple param sets used).
Do NOT use floating point. Do NOT write main.
Respond with a single complete C code block and CLOSE the fence with ```.
