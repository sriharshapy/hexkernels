Implement ONLY this function (Hexagon HVX C):
    void candidate_kernel(const int8_t *A, const int8_t *B,
                          const int32_t *bias,
                          const int8_t *gelu_lut,
                          int8_t *out,
                          int M, int N, int K);
Fused GEMM + bias + GELU-via-LUT -> int8:
  Step 1 -- matmul:  acc[i*N+j] = sum_k A[i*K+k]*B[k*N+j]   (int32 accumulator)
  Step 2 -- bias:    biased     = acc[i*N+j] + bias[j]        (int32; bias is per output column)
  Step 3 -- saturate: pre_lut  = saturate_to_int8(biased)     (clamp to [-128, 127])
  Step 4 -- GELU LUT: out[i*N+j] = gelu_lut[(uint8_t)(pre_lut + 128)]
A is [M x K] int8 row-major, B is [K x N] int8 row-major, bias is [N] int32.
gelu_lut is [256] int8; index into it as (uint8_t)(pre_lut + 128). Do NOT hardcode the LUT.
M=N=K=48. Bias is input data -- do NOT skip it.
Prefer HVX vrmpy for the K-reduction and vmem for the LUT gather.
Include <hexagon_types.h> and <hexagon_protos.h>. Do NOT write main().
Respond with a single complete C code block and CLOSE the fence with ```.
