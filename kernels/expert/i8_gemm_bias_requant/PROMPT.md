Implement ONLY this function (Hexagon HVX C):
    void candidate_kernel(const int8_t *A, const int8_t *B,
                          const int32_t *bias, int8_t *out,
                          int M, int N, int K,
                          int32_t mult, int shift, int8_t zp);
Fused GEMM + bias + requantize -> int8 (NO relu):
  Step 1 -- matmul: acc[i*N+j] = sum_k A[i*K+k]*B[k*N+j]   (int32 accumulator)
  Step 2 -- bias:   biased     = acc[i*N+j] + bias[j]        (int32; bias is per output column)
  Step 3 -- requantize to int8 (negatives are NOT clamped -- there is NO relu here):
    v    = (int64_t)biased * mult
    half = shift > 0 ? (1LL << (shift-1)) : 0
    r    = (v >= 0) ? (v + half) >> shift : -(((-v) + half) >> shift)
    r   += zp
    out[i*N+j] = saturate_to_int8(r)   // clamp to [-128, 127]
A is [M x K] int8 row-major, B is [K x N] int8 row-major, bias is [N] int32.
M=N=K=48. mult, shift, zp are runtime params -- do NOT hardcode them.
IMPORTANT: do NOT apply relu. Negative (acc+bias) values should be requantized as-is.
Prefer HVX vrmpy for the K-reduction. Include <hexagon_types.h> and <hexagon_protos.h>. Do NOT write main().
Respond with a single complete C code block and CLOSE the fence with ```.
