Implement ONLY this function (Hexagon HVX C):
    void candidate_kernel(const int8_t *A, const int8_t *B, const int8_t *residual,
                          int8_t *out, int M, int N, int K,
                          int32_t scale_mult, int scale_shift);

Compute an int8 matmul, requantize, then add a residual (skip-connection):
  A is [M x K] int8 row-major. B is [K x N] int8 row-major, CONVENTIONAL
  layout (B[k,j] = B[k*N+j]). residual is [M x N] int8. out is [M x N] int8.

  acc[i,j] = sum_k A[i*K+k] * B[k*N+j]                (int32 accumulate)
  r    = (int64_t)acc[i,j] * (int64_t)scale_mult
  half = (scale_shift > 0) ? (1LL << (scale_shift-1)) : 0
  q    = (r >= 0) ? ((r + half) >> scale_shift) : -((-r + half) >> scale_shift)
  requantized = clamp(q, -128, 127)
  out[i*N+j] = clamp((int32_t)requantized + (int32_t)residual[i*N+j], -128, 127)

IMPORTANT: the residual is added AFTER the requantize/shift step (its own
separate saturating add on top of the ALREADY-clamped int8 requantized
value) -- do NOT add it before the shift/requantize.

Pinned shapes: M=8, N=8, K=90 (90 = 22*4 + 2, NOT a multiple of 4 -- handle
the reduction tail). scale_mult and scale_shift are RUNTIME parameters
(anti-hardcode; at least 2 distinct pairs are swept). Prefer HVX vrmpy for
the K-reduction. Include <hexagon_types.h> and <hexagon_protos.h>. Do NOT
write main().
Respond with a single complete C code block and CLOSE the fence with ```.
