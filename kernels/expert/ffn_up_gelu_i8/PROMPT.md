Implement ONLY this function (Hexagon HVX C):
    void candidate_kernel(const int8_t *A, const int8_t *B,
                          const int32_t *bias, const int8_t *gelu_lut,
                          int8_t *out, int M, int N, int K);
Fused FFN "up" projection: GEMM + bias + GELU-via-LUT -> int8:
  Step 1 -- matmul:   acc[i,j] = sum_k A[i*K+k] * B[j*K+k]     (int32 accumulator)
  Step 2 -- bias:     biased   = acc[i,j] + bias[j]              (int32; bias is per output column)
  Step 3 -- saturate: pre      = saturate_to_int8(biased)         (clamp to [-128, 127])
  Step 4 -- GELU LUT: out[i,j] = gelu_lut[(uint8_t)(pre + 128)]
A is [M x K] int8 row-major.
B is [N x K] int8 row-major -- TRANSPOSED convention: row j of B is contiguous over K
  (this is NOT the [K x N] layout used by some sibling GEMM tasks).
bias is [N] int32. Bias is input data -- do NOT skip it.
gelu_lut is [256] int8; index into it as (uint8_t)(pre + 128). Do NOT hardcode the LUT --
  it is a runtime parameter swept across multiple variants by the harness.
M=8, N=8, K=90. K is NOT a multiple of 4 (90 = 22*4 + 2) -- the K-reduction has a
  2-element tail when processed in groups of 4.
The saturation before the LUT index is mandatory -- accumulator+bias values can exceed
  int8 range in either direction.
Prefer HVX vrmpy for the K-reduction. Include <hexagon_types.h> and <hexagon_protos.h>.
Do NOT write main(). Respond with a single complete C code block and CLOSE the fence with ```.
