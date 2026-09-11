Implement ONLY this function (Hexagon HVX C):
    void candidate_kernel(const int8_t *A, const int8_t *B,
                          const int32_t *bias, int8_t *out,
                          int M, int N, int K,
                          int32_t mult, int shift, int8_t zp);
Fused GEMM + bias + ReLU + requantize:
  Step 1 -- matmul: acc[i*N+j] = sum_k A[i*K+k]*B[k*N+j]   (int32 accumulator, i8*i8 products)
  Step 2 -- bias:   biased     = acc[i*N+j] + bias[j]        (int32; bias is per-column)
  Step 3 -- relu:   after_relu = max(biased, 0)
  Step 4 -- requantize to int8:
    v    = (int64_t)after_relu * mult
    half = shift > 0 ? (1LL << (shift-1)) : 0
    r    = (v >= 0) ? (v + half) >> shift : -(((-v) + half) >> shift)   // round half away from 0
    r   += zp
    out[i*N+j] = saturate_to_int8(r)   // clamp to [-128, 127]
A is [M x K] int8 row-major, B is [K x N] int8 row-major, bias is [N] int32, out is [M x N] int8.
M=N=K=64. mult, shift, zp are runtime params -- do NOT hardcode them.
Prefer HVX vrmpy for the K-reduction. Include <hexagon_types.h> and <hexagon_protos.h>. Do NOT write main().
Respond with a single complete C code block and CLOSE the fence with ```.
