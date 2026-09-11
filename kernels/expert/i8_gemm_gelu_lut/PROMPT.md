Implement ONLY this function (Hexagon HVX C):
 void candidate_kernel(const int8_t *A, const int8_t *B,
 const int8_t *gelu_lut,
 int8_t *out,
 int M, int N, int K);
Fused GEMM + GELU-via-LUT -> int8 (no bias step):
 Step 1 -- matmul: acc[i*N+j] = sum_k A[i*K+k]*B[k*N+j] (int32 accumulator)
 Step 2 -- saturate: pre_lut = saturate_to_int8(acc) (clamp to [-128, 127])
 Step 3 -- GELU LUT: out[i*N+j] = gelu_lut[(uint8_t)(pre_lut + 128)]
A is [M x K] int8 row-major, B is [K x N] int8 row-major.
gelu_lut is [256] int8; index it as (uint8_t)(pre_lut + 128). Do NOT hardcode the LUT.
M=N=K=48. There is NO bias step.
The saturation before the LUT index is mandatory -- accumulator values can exceed int8 range.
Respond with a single complete C code block and CLOSE the fence with ```.
