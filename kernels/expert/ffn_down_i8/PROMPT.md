Implement ONLY this function (Hexagon HVX C):
    void candidate_kernel(const int8_t *A, const int8_t *B, const int32_t *bias,
                          int8_t *out, int M, int N, int K,
                          int32_t scale_mult, int scale_shift);
Fused FFN "down" projection: GEMM + bias + requantize -> int8. There is NO activation
function here (this is the final FFN layer -- just bias then requantize):
  Step 1 -- matmul:    acc[i,j] = sum_k A[i*K+k] * B[j*K+k]      (int32 accumulator)
  Step 2 -- bias:      biased   = acc[i,j] + bias[j]               (int32; per output column)
  Step 3 -- requant (round-half-away-from-zero):
      r    = (int64_t)biased * (int64_t)scale_mult
      half = (scale_shift > 0) ? (1LL << (scale_shift-1)) : 0
      q    = (r >= 0) ? ((r + half) >> scale_shift) : -((-r + half) >> scale_shift)
      out[i,j] = clamp(q, -128, 127)
A is [M x K] int8 row-major (FFN hidden/intermediate activations, K = Dff dimension).
B is [N x K] int8 row-major -- TRANSPOSED convention: row j of B is contiguous over K
  (down-projection weight; NOT the [K x N] layout used by some sibling GEMM tasks).
bias is [N] int32. Bias is input data -- do NOT skip it.
scale_mult (int32, positive) and scale_shift (int, >=0) are RUNTIME parameters swept
  by the harness across multiple (mult, shift) pairs -- do NOT hardcode them.
M=8, N=8, K=110. K is NOT a multiple of 4 (110 = 27*4 + 2) -- the K-reduction has a
  2-element tail when processed in groups of 4.
Prefer HVX vrmpy for the K-reduction. Include <hexagon_types.h> and <hexagon_protos.h>.
Do NOT write main(). Respond with a single complete C code block and CLOSE the fence with ```.
