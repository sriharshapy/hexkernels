Implement ONLY this function (Hexagon HVX C):
    void candidate_kernel(const int8_t *A, const int8_t *B, int8_t *out,
                          int M, int N, int K,
                          int32_t zpA, int32_t scale_mult, int scale_shift);

A is [M x K] int8 row-major, a quantized ACTIVATION with a runtime int32
zero-point zpA to subtract. B is [K x N] int8 row-major, CONVENTIONAL
layout (B[k,j] = B[k*N+j]) -- the WEIGHT, already zero-centered (no
zero-point on B). out is [M x N] int8, requantized.

Use the STANDARD zero-point GEMM decomposition -- do NOT compute
(A[i,k]-zpA) as an int8 before multiplying (it can overflow int8 range
since zpA can push values outside [-128,127]):
  rawdot[i,j] = sum_k A[i*K+k] * B[k*N+j]         (int32; plain int8 x int8)
  colsum[j]   = sum_k B[k*N+j]                     (int32; computed ONCE,
                                                     shared across all i)
  acc[i,j]    = rawdot[i,j] - zpA * colsum[j]
  r    = (int64_t)acc[i,j] * (int64_t)scale_mult
  half = (scale_shift > 0) ? (1LL << (scale_shift-1)) : 0
  q    = (r >= 0) ? ((r + half) >> scale_shift) : -((-r + half) >> scale_shift)
  out[i*N+j] = clamp(q, -128, 127)

Pinned shapes: M=8, N=8, K=94 (94 = 23*4 + 2, NOT a multiple of 4 --
handle the reduction tail). zpA, scale_mult, scale_shift are RUNTIME
parameters (anti-hardcode; several (zpA, scale_mult, scale_shift) triples
are swept, including zpA=0 which must degenerate exactly to a plain GEMM,
and both a negative and a positive zpA). Do NOT hardcode any of them.
Prefer HVX vrmpy for the K-reduction. Include <hexagon_types.h> and
<hexagon_protos.h>. Do NOT write main().
Respond with a single complete C code block and CLOSE the fence with ```.
