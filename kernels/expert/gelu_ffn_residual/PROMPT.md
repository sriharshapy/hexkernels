Implement ONLY this function (Hexagon HVX C):
    void candidate_kernel(const int8_t   *x,
                          const int8_t   *W1, const int32_t *b1,
                          const int8_t   *gelu_lut,
                          const int8_t   *W2, const int32_t *b2,
                          const int8_t   *x_res,
                          const int8_t   *gamma, const int8_t *beta,
                          const uint8_t  *inv_lut,
                          int8_t         *out,
                          int M, int K, int V, int D);

FFN with GELU + residual add + LayerNorm (all integer):
  Layer 1 -- GEMM + bias + sat8 + GELU LUT:
    hid[i*V+v] = gelu_lut[(uint8_t)(sat8(sum_k x[i*K+k]*W1[k*V+v] + b1[v]) + 128)]
    DO NOT hardcode the LUT -- it is a runtime parameter.

  Layer 2 -- GEMM + bias + residual add:
    ffn[i*D+j] = sum_v hid[i*V+v]*W2[v*D+j] + b2[j]    (int32)
    res[i*D+j] = sat8(ffn[i*D+j] + (int32_t)x_res[i*D+j])

  LayerNorm per token i:
    mu       = mean(res[i*D+0..D-1])            (integer divide by D)
    var      = mean((res[i*D+j]-mu)^2)          (integer divide by D)
    v_idx    = clamp(var, 0, 255)
    inv      = inv_lut[v_idx]                   (runtime LUT -- NOT hardcoded)
    for j in [0,D):
      scaled = (( res[i*D+j]-mu ) * gamma[j] + 64) >> 7
      normed = (scaled * inv + 128) >> 8
      out[i*D+j] = sat8(normed + beta[j])

Dims: M=8, K=16, V=32, D=16. All LUTs (gelu_lut, inv_lut), gamma, beta are runtime params.
Prefer HVX vrmpy for GEMM K-reductions. Include <hexagon_types.h> and <hexagon_protos.h>. Do NOT write main().
Respond with a single complete C code block and CLOSE the fence with ```.
